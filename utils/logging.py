import torch
import wandb
import numpy as np
import random
import os
import sys
import warnings
import yaml
from omegaconf import OmegaConf


# Sanity checks
def sanity_checks(cfg, max_sample_length):
    """
    Basic sanity checks for model configuration and data compatibility
    """

    # Check if vocabulary size and sequence length are compatible
    assert(cfg.model.context_size >= max_sample_length)
    assert(cfg.model.n_embd % cfg.model.n_head == 0)

# Seed
def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    # rng = np.random.default_rng(seed)
    # true_seed = int(rng.integers(2**30))

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Wandb and logging
def open_log(cfg):
    """
    Open log file and redirect stdout and stderr to it
    """
    print(cfg)
    os.makedirs('logs/' + cfg.tag, exist_ok=True)
    if cfg.deploy:
        fname = 'logs/' + cfg.tag + '/' + wandb.run.id + ".log"
        fout = open(fname, "a", 1)
        sys.stdout = fout
        sys.stderr = fout
        print(cfg)
        return fout


def save_config(cfg):
    """
    Save configuration to file
    """
    if cfg.deploy and wandb.run is not None:
        results_dir = 'sae_results/' + cfg.tag + "/" + wandb.run.id
    else:
        # Use timestamp when wandb is disabled
        import time
        timestamp = str(int(time.time()))
        results_dir = 'sae_results/' + cfg.tag + "/" + timestamp
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + '/conf.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg), f)


def init_wandb(cfg, project_name):
    """
    Initialize wandb
    """
    if cfg.deploy:

        model_name = 'llama' if 'Llama' in cfg.llm.model_hf_name else 'gemma'

        # Create a meaningful run name based on key hyperparameters
        try:
            if cfg.sae.sae_diff_type == 'relu':
                run_name = f"temporal_{model_name}_{cfg.sae.sae_diff_type}_gamma{cfg.sae.gamma_reg}"
            elif cfg.sae.sae_diff_type == 'topk':
                run_name = f"temporal_{model_name}_{cfg.sae.sae_diff_type}_K{cfg.sae.kval_topk}"
            elif cfg.sae.sae_diff_type == 'nullify':
                run_name = f"temporal_{model_name}_{cfg.sae.sae_diff_type}"
            elif cfg.sae.sae_diff_type == 'batchtopk':
                run_name = f"temporal_{model_name}_{cfg.sae.sae_diff_type}_K{cfg.sae.kval_topk}"
            else:
                raise ValueError(f"Invalid sae_diff_type: {cfg.sae.sae_diff_type}")
        except:
            if cfg.sae.sae_type == 'relu':
                run_name = f"standard_{model_name}_{cfg.sae.sae_type}_gamma{cfg.sae.gamma_reg}"
            elif cfg.sae.sae_type == 'topk' or cfg.sae.sae_type == 'batchtopk':
                run_name = f"standard_{model_name}_{cfg.sae.sae_type}_K{cfg.sae.kval_topk}"
            elif cfg.sae.sae_type == 'MP':
                run_name = f"standard_{model_name}_{cfg.sae.sae_type}_K{cfg.sae.mp_kval}"
            elif cfg.sae.sae_type == 'SpaDE':
                run_name = f"standard_{model_name}_{cfg.sae.sae_type}_gamma{cfg.sae.gamma_reg}"
            else:
                raise ValueError(f"Invalid sae_type: {cfg.sae.sae_type}")
        wandb.init(project=project_name, name=run_name)
        wandb.config.update(OmegaConf.to_container(cfg))


def cleanup(cfg, fp):
    """
    Close log file and wandb
    """
    if cfg.deploy:
        fp.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        wandb.finish()


def log_train(it, deploy, lr, train_loss, train_lengths):
    """
    Log training information
    """
    if deploy and len(train_loss) > 0:
        wandb.log({
            "train": {k: np.mean(v) for k, v in train_loss.items()},
            "iteration": it,
            "lr": lr
            })

        for k, v in train_lengths.items():
            wandb.log({'train': {f'lengths/{k}': v}})

    print("train -- iter: %d, lr: %.6f, loss: %.4f" % (it, lr, np.mean(train_loss['total'])))
    train_loss = {k: [] for k in train_loss.keys()}
    return train_loss


def log_eval(deploy, it, save_tables, grammaticality_results):
    """
    Log eval information
    """

    if deploy:
        wandb.log({'eval': {'iteration': it}})

        # Grammaticality
        if grammaticality_results is not None:
            for key in grammaticality_results.keys():
                if key == 'failures':
                    continue

                elif key == 'validity':
                    wandb.log({'grammaticality': {'validity': grammaticality_results['validity']}})

                else:
                    for k, v in grammaticality_results[key].items():
                        wandb.log({'grammaticality': {f'{key} ({k})': v}})

    print("eval -- iter: %d" % it)

    return save_tables+1


def log_sae_train(it, deploy, lr, train_log):
    """
    Log training information
    """
    if deploy and len(train_log['mse']) > 0:
        wandb.log({
            "MSE": np.mean(train_log['mse']),
            "NMSE": np.mean(train_log['nmse']),
            "MSE Bias": np.mean(train_log['mse_bias']),
            "MSE Pred": np.mean(train_log['mse_pred']),
            "NMSE Pred": np.mean(train_log['nmse_pred']),
            "MSE Novel": np.mean(train_log['mse_novel']),
            "NMSE Novel": np.mean(train_log['nmse_novel']),
            "Relative Energy in Pred": np.mean(train_log['rel_energy']),
            "Pred-Novel Recons Similarity": np.mean(train_log['pn_recons_similarity']),
            "Pred-Novel Code Similarity": np.mean(train_log['pn_code_similarity']),
            "Error Similarity": np.mean(train_log['error_similarity']),
            "Variance_explained": np.mean(train_log['variance_explained']),
            "Reg.": np.mean(train_log['reg']),
            "% Sparsity": np.mean(train_log['p_sparsity']),
            "Lambda": np.mean(train_log['lambda']),
            "iteration": it,
            "lr": lr
            })

    print("train -- iter: %d, lr: %.6f, loss: %.4f" % (it, lr, np.mean(train_log['mse'])))
    train_log = {
        'mse': [],
        'nmse': [],
        'mse_bias': [],
        'mse_pred': [],
        'nmse_pred': [],
        'mse_novel': [],
        'nmse_novel': [],
        'rel_energy': [],
        'pn_recons_similarity': [],
        'pn_code_similarity': [],
        'error_similarity': [],
        'variance_explained': [],
        'reg': [],
        'p_sparsity': [],
        'lambda': [],
    }

    return train_log


# Save model
def save_model(cfg, net, optimizer, it):
    """
    Save model checkpoint
    """
    if cfg.deploy:
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter': it,
            'config': cfg,
        }
        fdir = 'results/' + cfg.tag + "/" + wandb.run.id
        os.makedirs(fdir, exist_ok=True)
        if cfg.log.save_multiple:
            fname = os.path.join(fdir, 'ckpt_' + str(it+1) + '.pt')
        else:
            fname = os.path.join(fdir, 'latest_ckpt.pt')
        torch.save(checkpoint, fname)
        return fdir


# Save model
def save_sae(cfg, sae, optimizer, it):
    """
    Save model checkpoint
    """
    if cfg.deploy:
        checkpoint = {
            'sae': sae.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter': it,
            'config': cfg,
        }
        fdir = 'sae_results/' + cfg.tag + "/" + wandb.run.id
        os.makedirs(fdir, exist_ok=True)
        if cfg.log.save_multiple:
            fname = os.path.join(fdir, 'ckpt_' + str(it+1) + '.pt')
        else:
            fname = os.path.join(fdir, 'latest_ckpt.pt')
        torch.save(checkpoint, fname)
        return fdir

