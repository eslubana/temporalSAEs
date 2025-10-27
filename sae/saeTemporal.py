import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from sae.utils import ManualAttention


class TemporalSAE(torch.nn.Module):
    def __init__(self, dimin=2, width=5, n_heads=8, sae_diff_type='relu', kval_topk=None, tied_weights=True,
        n_attn_layers=1, bottleneck_factor=64, inference_mode_batchtopk=False, 
        min_act_regularizer_batchtopk=0.999):
        """
        dimin: (int)
            input dimension
        width: (int)
            width of the encoder
        n_heads: (int)
            number of attention heads
        sae_diff_type: (str)
            type of sae to express the per-token difference
        kval_topk: (int)
            k in topk sae_diff_type
        n_attn_layers: (int)
            number of attention layers
        inference_mode_batchtopk: (bool)
            whether to use inference mode for batchtopk
        min_act_regularizer_batchtopk: (float)
            exponential moving average weight for batchtopk threshold
        """
        super(TemporalSAE, self).__init__()
        self.sae_type = 'temporal'
        self.width = width
        self.dimin = dimin
        self.eps = 1e-6
        self.lam = 1 / (4 * dimin)
        self.tied_weights = tied_weights

        ## Attention parameters
        self.n_attn_layers = n_attn_layers
        self.attn_layers = nn.ModuleList([
            ManualAttention(dimin=width, 
                            n_heads=n_heads, 
                            bottleneck_factor=bottleneck_factor, 
                            bias_k=True, bias_q=True, bias_v=True, bias_o=True)
            for _ in range(n_attn_layers)
            ])

        ## Dictionary parameters
        self.D = nn.Parameter(torch.randn((width, dimin))) # N(0,1) init
        self.b = nn.Parameter(torch.zeros((1, dimin)))
        if not tied_weights:
            self.E = nn.Parameter(torch.randn((dimin, width))) # N(0,1) init

        ## SAE-specific parameters
        self.sae_diff_type = sae_diff_type
        self.kval_topk = kval_topk if sae_diff_type in ['topk', 'batchtopk'] else None

        ## BatchTopK-specific parameters
        if sae_diff_type == 'batchtopk':
            self.inference_mode_batchtopk = inference_mode_batchtopk
            self.expected_min_act = nn.Parameter(torch.zeros(1))
            self.expected_min_act.requires_grad = False
            self.min_act_regularizer_batchtopk = min_act_regularizer_batchtopk

    def forward(self, x_input, return_graph=False, inf_k=None):
        B, L, _ = x_input.size()
        E = self.D.T if self.tied_weights else self.E

        ### Define context and target ###
        x_input = x_input - self.b

        ### Tracking variables ###
        attn_graphs = []

        ### Predictable part ###
        z_pred = torch.zeros((B, L, self.width), device=x_input.device, dtype=x_input.dtype)
        for attn_layer in self.attn_layers:

            z_input = F.relu(torch.matmul(x_input * self.lam, E)) # [batch, length, width]
            z_ctx = torch.cat((torch.zeros_like(z_input[:, :1, :]), z_input[:, :-1, :].clone()), dim=1) # [batch, length, width]

            # Compute codes using attention
            z_pred_, attn_graphs_ = attn_layer(z_ctx, z_input, get_attn_map=return_graph)

            # Take back to input space
            z_pred_ = F.relu(z_pred_)
            Dz_pred_ = torch.matmul(z_pred_, self.D)
            Dz_norm_ = (Dz_pred_.norm(dim=-1, keepdim=True) + self.eps)

            # Compute projection
            proj_scale = (Dz_pred_ * x_input).sum(dim=-1, keepdim=True) / Dz_norm_.pow(2)

            # Add the projection to the reconstructed 
            z_pred = z_pred + (z_pred_ * proj_scale)

            # Remove the projection from the input
            x_input = x_input - proj_scale * Dz_pred_ # [batch, length, width]

            # Add the attention graph if return_graph is True
            if return_graph:
                attn_graphs.append(attn_graphs_)

        ### Novel part (identified using the residual target signal) ###
        if self.sae_diff_type=='relu':
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))

        elif self.sae_diff_type=='topk':
            kval = self.kval_topk if inf_k is None else inf_k
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))
            _, topk_indices = torch.topk(z_novel, kval, dim=-1)
            mask = torch.zeros_like(z_novel)
            mask.scatter_(-1, topk_indices, 1)
            z_novel = z_novel * mask

        elif self.sae_diff_type=='batchtopk':
            kval = self.kval_topk if inf_k is None else inf_k
            kval_full_batch = kval * B * L  # Total activations across batch and sequence
            
            # Encode
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))
            
            # Sparsify
            if not self.inference_mode_batchtopk:
                # Do batch top k sparsification during training
                z_flat = z_novel.flatten()
                topk_values, topk_indices = torch.topk(z_flat, kval_full_batch, dim=-1)
                z_novel_sparse = torch.zeros_like(z_flat)
                z_novel_sparse.scatter_(-1, topk_indices, topk_values)
                z_novel = z_novel_sparse.reshape(z_novel.shape)
                
                # Update moving average of min activations for thresholding during inference
                active = z_flat[z_flat > 0]  # Get all positive activations
                if active.size(0) == 0:
                    min_activation = 0.0
                else:
                    min_activation = active.min().detach().to(dtype=z_novel.dtype)
                
                # Exponential moving average update
                self.expected_min_act[0] = (self.min_act_regularizer_batchtopk * self.expected_min_act[0]) + (
                    (1 - self.min_act_regularizer_batchtopk) * min_activation
                )
            else:
                # Do threshold-based sparsification during inference
                z_novel = z_novel * (z_novel > self.expected_min_act[0])

        elif self.sae_diff_type=='nullify':
            z_novel = torch.zeros_like(z_pred)

        ### Reconstruction ###
        x_recons = torch.matmul(z_novel + z_pred, self.D) + self.b # [batch, length, dimin]

        ### Compute the predicted vs. novel reconstructions, sans the bias (allows to check context / dictionary's value) ###
        with torch.no_grad():
            x_pred_recons = torch.matmul(z_pred, self.D)
            x_novel_recons = torch.matmul(z_novel, self.D)

        ### Return the dictionary ###
        results_dict = {
            'novel_codes': z_novel, 
            'novel_recons': x_novel_recons,
            'pred_codes': z_pred,
            'pred_recons': x_pred_recons,
            'attn_graphs': torch.stack(attn_graphs, dim=1) if return_graph else None
            }

        return x_recons, results_dict

    @classmethod
    def from_pretrained(cls, folder_path, dtype, device, **kwargs):
        """
        Load a pretrained TemporalSAE from a folder containing conf.yaml and latest_ckpt.pt

        Args:
            folder_path: Path to folder containing conf.yaml and latest_ckpt.pt
            dtype: Target dtype for the model
            device: Target device for the model
            **kwargs: Override any config parameters
        """
        # Load config from yaml file
        config_path = os.path.join(folder_path, 'conf.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract model parameters from config
        model_args = {
            'dimin': config['llm']['dimin'],
            'width': int(config['llm']['dimin'] * config['sae']['exp_factor']),
            'n_heads': config['sae']['n_heads'],
            'sae_diff_type': config['sae']['sae_diff_type'],
            'kval_topk': config['sae']['kval_topk'],
            'tied_weights': config['sae']['tied_weights'],
            'n_attn_layers': config['sae']['n_attn_layers'],
            'bottleneck_factor': config['sae']['bottleneck_factor'],
            'inference_mode_batchtopk': config['sae'].get('inference_mode_batchtopk', False),
            'min_act_regularizer_batchtopk': config['sae'].get('min_act_regularizer_batchtopk', 0.999),
        }

        # Override with any provided kwargs
        model_args.update(kwargs)

        # Create the model
        autoencoder = cls(**model_args)

        # Load the checkpoint
        ckpt_path = os.path.join(folder_path, 'latest_ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Load the state dict
        if 'sae' in checkpoint:
            autoencoder.load_state_dict(checkpoint['sae'])
        else:
            autoencoder.load_state_dict(checkpoint)

        autoencoder = autoencoder.to(device=device, dtype=dtype)
        return autoencoder