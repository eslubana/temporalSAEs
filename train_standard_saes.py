import hydra
import torch
import torch.nn.functional as F

from sae import SAEStandard, step_fn

from utils import init_wandb, set_seed, save_config, open_log, cleanup
from utils import update_cosine_warmup_lr
from utils import save_sae, log_sae_train

from precompute_llm_activations.cache_utils import ActivationPrecomputedBuffer, dtype_str_to_torch
from tqdm import tqdm



# @hydra.main(config_path="./config/sae_train", config_name="standard_conf_llama.yaml", version_base="1.3")
@hydra.main(config_path="./config/sae_train", config_name="standard_conf.yaml", version_base="1.3")
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
    sae = SAEStandard(
        dimin=cfg.llm.dimin,
        width=int(cfg.llm.dimin * cfg.sae.exp_factor),
        sae_type=cfg.sae.sae_type,
        kval_topk=cfg.sae.kval_topk,
        mp_kval=cfg.sae.mp_kval,
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
    scaling_factor = 1 / 236.184564 if sae.sae_type == 'SpaDE' else cfg.sae.scaling_factor

    for _ in training_iterator:

        # if it > 1000000:  # Intentional break, in case we want to stop training
        #     save_sae(cfg, sae, optimizer, it)
        #     break

        # Get activations from buffer
        activations = next(activation_buffer)
        batch_size, seq_len, hidden_size = activations.size()
        activations = activations.to(device)
        activations = activations.view(batch_size * seq_len, hidden_size) * scaling_factor

        # Update LR
        it, lr = update_cosine_warmup_lr(
            it, cfg.optimizer, optimizer, cfg.data.num_total_steps
        )

        # Loss computation
        optimizer.zero_grad(set_to_none=True)  # Set gradients to None
        with torch.amp.autocast(device_type="cuda", dtype=dt):  # Mixed precision

            ## Get SAE output
            recons_activations, latent_code = sae(activations, return_hidden=True)

            # Flatten activations and predictions
            recons_activations = recons_activations.view(batch_size * seq_len, hidden_size)

            ## Regularization loss
            # ReLU: L1 
            if sae.sae_type == 'relu':
                reg = torch.norm(latent_code, p=1, dim=-1).mean()

            # JumpReLU: L0 loss
            elif sae.sae_type == 'jumprelu':
                bandwidth = 1e-3
                reg = torch.mean(torch.sum(
                    step_fn(latent_code, torch.exp(sae.logthreshold), bandwidth), 
                    dim=-1))

            # TopK: No regularization
            elif sae.sae_type == 'topk' or sae.sae_type == 'batchtopk' or sae.sae_type == 'MP':
                reg = torch.tensor([0.0], device=device)

            # SpaDE: Sparsemax with Distance based penalty
            elif sae.sae_type == 'SpaDE':

                # Optimized using kernel trick: ||x - a||^2 = ||x||^2 + ||a||^2 - 2*x^T*a
                act_norm_sq = torch.sum(activations.pow(2), dim=-1, keepdim=True)  # (batch_size, 1)
                ae_norm_sq = torch.sum(sae.Ae.pow(2), dim=-1, keepdim=True).T  # (1, width)
                dot_product = torch.matmul(activations, sae.Ae.T)  # (batch_size, width)
                dist_penalty_encoder = act_norm_sq + ae_norm_sq - 2 * dot_product
                reg = (dist_penalty_encoder * latent_code).sum(dim=-1).mean()

            else:
                raise ValueError('Invalid SAE type')

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
                train_log["reg"].append(reg.item())  # Reg loss
                p_sparsity = (
                    (latent_code.abs() < 1e-5).sum()
                    / latent_code.numel() * 100
                )  # Sparsity of the latent code
                train_log["p_sparsity"].append(p_sparsity.item())  # Sparsity
                train_log['lambda'].append(sae.lambda_val.data.item()) # Lambda value

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
