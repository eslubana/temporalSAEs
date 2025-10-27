import hydra
import torch
import torch.nn.functional as F

from sae import TemporalSAE

from utils import init_wandb, set_seed, save_config, open_log, cleanup
from utils import update_cosine_warmup_lr
from utils import save_sae, log_sae_train

from precompute_llm_activations.cache_utils import ActivationPrecomputedBuffer, dtype_str_to_torch
from tqdm import tqdm



@hydra.main(config_path="./config/sae_train", config_name="temporal_conf_llama.yaml", version_base="1.3")
# @hydra.main(config_path="./config/sae_train", config_name="temporal_conf.yaml", version_base="1.3")
def main(cfg):

    # Setup Environment
    set_seed(cfg.seed)
    device = cfg.device_id
    init_wandb(cfg, project_name=cfg.log.wandb_project_name)
    save_config(cfg)
    fp = open_log(cfg)

    # Create precomputed activation buffer
    activation_buffer = ActivationPrecomputedBuffer(
        cfg.data.cache_dir, cfg.data.context_length, cfg.data.batch_size
    )

    # Define SAE
    sae = TemporalSAE(
        dimin=cfg.llm.dimin,
        width=int(cfg.llm.dimin * cfg.sae.exp_factor),
        n_heads=cfg.sae.n_heads,
        sae_diff_type=cfg.sae.sae_diff_type,
        kval_topk=cfg.sae.kval_topk,
        tied_weights=cfg.sae.tied_weights,
        n_attn_layers=cfg.sae.n_attn_layers,
        bottleneck_factor=cfg.sae.bottleneck_factor,
    ).to(device)
    sae.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )

    # Train
    train(cfg, sae, optimizer, activation_buffer, device)

    # Cleanup
    activation_buffer.cleanup()

    # Close wandb and log file
    cleanup(cfg, fp)


def train(cfg, sae, optimizer, activation_buffer, device):
    """
    Training function
    """

    # Some hparams variables
    lr, it = 0.0, 0
    dt = dtype_str_to_torch[cfg.data.dtype]

    # Stuff to do only on the first GPU
    print("Total training steps: ", cfg.data.num_total_steps)
    print("Learning rate warmup steps: ", cfg.optimizer.warmup_iters)

    # Save initial SAE
    save_sae(cfg, sae, optimizer, it)

    # Initialize SAE training log
    train_log = {
        "variance_explained": [],
        "mse": [],
        "nmse": [],
        "mse_bias": [],
        "mse_pred": [],
        "nmse_pred": [],
        "mse_novel": [],
        "nmse_novel": [],
        "reg": [],
        "pn_recons_similarity": [],
        "pn_code_similarity": [],
        "error_similarity": [],
        "rel_energy": [],
        "p_sparsity": [],
        "lambda": [],
    }

    # Create tqdm progress bar
    training_iterator = range(cfg.data.num_total_steps)
    training_iterator = tqdm(training_iterator, desc="Training Progress", unit="step")

    for _ in training_iterator:

        # if it > 1000000:  # Intentional break, in case we want to stop training
        #     save_sae(cfg, sae, optimizer, it)
        #     break

        # Get activations from buffer
        activations = next(activation_buffer)
        batch_size, seq_len, hidden_size = activations.size()
        activations = activations.to(device) * cfg.sae.scaling_factor

        # Update LR
        it, lr = update_cosine_warmup_lr(
            it, cfg.optimizer, optimizer, cfg.data.num_total_steps
        )

        # Loss computation
        optimizer.zero_grad(set_to_none=True)  # Set gradients to None
        with torch.amp.autocast(device_type="cuda", dtype=dt):  # Mixed precision

            ## Get SAE output
            recons_activations, intermediates_dict = sae(activations, return_graph=False)

            # Flatten activations and predictions
            activations = activations.view(batch_size * seq_len, hidden_size)
            recons_activations = recons_activations.view(batch_size * seq_len, hidden_size)
            pred_activations = intermediates_dict["pred_recons"].view(
                batch_size * seq_len, hidden_size
            )
            pred_code = intermediates_dict["pred_codes"].view(batch_size * seq_len, -1)
            novel_activations = intermediates_dict["novel_recons"].view(
                batch_size * seq_len, hidden_size
            )
            novel_code = intermediates_dict["novel_codes"].view(batch_size * seq_len, -1)

            ## Loss
            if cfg.sae.sae_diff_type == "relu":
                reg = novel_code.norm(dim=1).mean()
            elif cfg.sae.sae_diff_type == "topk" or cfg.sae.sae_diff_type == "nullify":
                reg = torch.tensor(0.0)
            else:
                raise ValueError(f"Unknown SAE diff type: {cfg.sae.sae_diff_type}")

            ## MSE
            loss = (
                cfg.sae.gamma_reg * reg
                + F.mse_loss(recons_activations, activations, reduction="sum")
                / activations.shape[0]
            )  # Normalize by number of tokens

            ## Update model
            loss.backward()  # Compute gradients
            if cfg.optimizer.grad_clip > 0.0:  # Gradient clipping
                torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.optimizer.grad_clip)
            optimizer.step()  # Update weights

            ## Logging
            with torch.no_grad():
                # Log MSE / variance explained
                av_norm_activations = activations.norm(dim=-1).pow(2).mean().item()
                train_log['mse'].append(loss.item()) # MSE loss
                train_log['nmse'].append(loss.item() / av_norm_activations) # Normalized MSE loss
                train_log['variance_explained'].append((1 - (recons_activations - activations).var() / activations.var()).item())
                
                ## Some metrics
                # Bias loss
                mse_bias = (
                    F.mse_loss(
                        activations, sae.b.expand(activations.size(0), -1), reduction="sum"
                    )
                    / activations.shape[0]
                )

                # Relative energy of the predictable part
                rel_energy = pred_activations.norm(dim=-1).pow(2).mean() / (
                    pred_activations.norm(dim=-1).pow(2).mean()
                    + novel_activations.norm(dim=-1).pow(2).mean()
                )

                # Similarity between the predictable and the novel part
                pn_code_similarity = F.cosine_similarity(
                    pred_code, novel_code, dim=-1
                ).mean()

                # Similarity between recons from predictable and novel part
                pn_recons_similarity = F.cosine_similarity(
                    pred_activations, novel_activations, dim=-1
                ).mean()

                # Normalize activations
                activations = activations - sae.b
                av_norm_activations = activations.norm(dim=-1).pow(2).mean().item()

                # Novel loss
                mse_novel = (
                    F.mse_loss(novel_activations, activations, reduction="sum")
                    / activations.shape[0]
                )

                # Predictable loss
                mse_pred = (
                    F.mse_loss(pred_activations, activations, reduction="sum")
                    / activations.shape[0]
                )

                # Similarity between error from predictable and novel part
                error_similarity = F.cosine_similarity(
                    pred_activations - activations, novel_activations - activations, dim=-1
                ).mean()

                # Sparsity of the novel part
                p_sparsity = (
                    (intermediates_dict["novel_codes"].abs() < 1e-5).sum()
                    / intermediates_dict["novel_codes"].numel()
                    * 100
                )  # Sparsity of the novel part

                # Log metrics
                train_log["mse_bias"].append(mse_bias.item())  # MSE over the bias part
                train_log["mse_pred"].append(
                    mse_pred.item()
                )  # MSE over the predictable part
                train_log["nmse_pred"].append(
                    mse_pred.item() / av_norm_activations
                )  # Normalized MSE over the predictable part
                train_log["mse_novel"].append(mse_novel.item())  # MSE over the novel part
                train_log["nmse_novel"].append(
                    mse_novel.item() / av_norm_activations
                )  # Normalized MSE over the novel part
                train_log["pn_code_similarity"].append(
                    pn_code_similarity.item()
                )  # Similarity between the predictable and the novel part
                train_log["pn_recons_similarity"].append(
                    pn_recons_similarity.item()
                )  # Similarity between the predictable and the novel part
                train_log["error_similarity"].append(
                    error_similarity.item()
                )  # Similarity between the predictable and the novel part
                train_log["rel_energy"].append(
                    rel_energy.item()
                )  # Relative energy of the predictable part
                train_log["reg"].append(reg.item())  # Reg loss
                train_log["p_sparsity"].append(p_sparsity.item())  # Sparsity

        # Log train metrics
        if it % cfg.log.log_interval == 0:
            train_log = log_sae_train(it, cfg.deploy, lr, train_log)

        # Save model every few iterations
        if it % cfg.log.save_interval == 0:
            save_sae(cfg, sae, optimizer, it)

    # Save one last time
    save_sae(cfg, sae, optimizer, it)


if __name__ == "__main__":
    main()  # This runs Hydra first and passes cfg properly
