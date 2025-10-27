import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax
import torch
import yaml
import os
from sae.utils import softplus_inverse, jumprelu

class SAEStandard(torch.nn.Module):
    def __init__(self, dimin=2, width=5, sae_type='relu', kval_topk=None, 
        mp_kval=None, lambda_init=None, inference_mode_batchtopk=False,
        min_act_regularizer_batchtopk = 0.999):
        """
        dimin: (int)
            input dimension
        width: (int)
            width of the encoder
        sae_type: (str)
            one of 'relu', 'topk', 'jumprelu'
        kval_topk: (int)
            k in topk sae_type
        """
        super(SAEStandard, self).__init__()
        self.sae_type = sae_type
        self.width = width
        self.dimin = dimin
        self.eps = 1e-6

        ## Encoder parameters
        self.Ae = nn.Parameter(torch.randn((width, dimin))) #N(0,1) init
        if sae_type != 'MP' and sae_type != 'mp':
            self.be = nn.Parameter(torch.zeros((1, width)))

        ## Decoder parameters
        if sae_type != 'MP' and sae_type != 'mp':
            self.Ad = nn.Parameter(torch.randn((dimin, width))) #N(0,1) init
            with torch.no_grad():
                self.Ad.copy_(self.Ae.T) #at init, decoder is the transpose of encoder
        self.bd = nn.Parameter(torch.zeros((1, dimin)))

        ## Parameters for specific SAEs
        # JumpReLU
        if sae_type=='jumprelu':
            self.logthreshold = nn.Parameter(torch.log(1e-3*torch.ones((1, width))))
            self.bandwidth = 1e-3 #width of rectangle used in approx grad of jumprelu wrt threshold

        lambda_init = 1/(4*dimin) if lambda_init is None else lambda_init
        lambda_pre = softplus_inverse(lambda_init)
        self.lambda_pre = nn.Parameter(lambda_pre, requires_grad=False) #not trainable
        
        # Topk parameter
        if sae_type=='topk':
            if kval_topk is not None:
                self.kval_topk = kval_topk

        # BatchTopK parameter
        if sae_type=='batchtopk':
            """Batch-top-K is differing in two aspects from standard topk:
                1. Training: Take top n * k activations across the batch of n samples. Monitor the expected minimum activation per latent.
                2. Inference: Apply thresholding by expected minimum activation per latent, effectively applying Jump Relu.
                   (We want the ability to encode activations of single tokens, so can’t apply batch thresholding here.)
            """
            self.inference_mode_batchtopk = inference_mode_batchtopk
            self.expected_min_act = nn.Parameter(torch.zeros(1))
            self.expected_min_act.requires_grad = False
            if kval_topk is not None:
                self.kval_topk = kval_topk
            self.min_act_regularizer_batchtopk = min_act_regularizer_batchtopk

        # MP parameter
        if sae_type=='MP' or sae_type=='mp':
            if mp_kval is not None:
                self.mp_kval = mp_kval

        # SpaDE
        if sae_type=='SpaDE':
            lambda_init = 1/(width*dimin) if lambda_init is None else lambda_init
            lambda_pre = softplus_inverse(lambda_init)
            self.lambda_pre = nn.Parameter(lambda_pre) #trainable parameter (~inv temp) for sparsemax

            # Normalize the encoder and decoder weights
            with torch.no_grad():
                Ae_unit = self.Ae / (self.eps + torch.linalg.norm(self.Ae, dim=1, keepdim=True))
                self.Ae.copy_(Ae_unit)
                Ad_unit = self.Ad / (self.eps + torch.linalg.norm(self.Ad, dim=1, keepdim=True)) * 48.0
                self.Ad.copy_(Ad_unit)

    @property
    def lambda_val(self): #lambda_val is lambda, forced to be positive here
        return F.softplus(self.lambda_pre)


    def forward(self, x, return_hidden=False, inf_k=None, return_graph=None):
        lam = self.lambda_val

        if self.sae_type=='relu':
            x = x-self.bd #pre-encoder bias
            x = torch.matmul(x, self.Ae.T) + self.be
            codes = F.relu(lam*x)
            x = torch.matmul(codes, self.Ad.T) + self.bd

        elif self.sae_type=='topk':
            kval = self.kval_topk if inf_k is None else inf_k
            x = x-self.bd
            x = torch.matmul(x, self.Ae.T)
            topk_values, topk_indices = torch.topk(F.relu(x), kval, dim=-1)
            codes = torch.zeros_like(x)
            codes.scatter_(-1, topk_indices, topk_values)
            codes *= lam
            x = torch.matmul(codes, self.Ad.T) + self.bd

        elif self.sae_type=="batchtopk":
            kval = self.kval_topk if inf_k is None else inf_k
            kval_full_batch = kval * x.shape[0]

            # Encode
            x = x-self.bd
            x = torch.matmul(x, self.Ae.T)

            # Sparsify
            if not self.inference_mode_batchtopk:
                # Do batch top k sparsification during training
                x_flat = x.flatten()
                topk_values, topk_indices = torch.topk(F.relu(x_flat), kval_full_batch, dim=-1)
                codes = torch.zeros_like(x_flat)
                codes.scatter_(-1, topk_indices, topk_values)
                codes = codes.reshape(x.shape)
                codes *= lam

                # Update moving average of min activations for thresholding during inference
                active = x_flat[x_flat > 0]  # Get all positive activations
                if active.size(0) == 0:
                    min_activation = 0.0
                else:
                    min_activation = active.min().detach().to(dtype=x.dtype)  # min of positive activations

                # Exponential moving average update
                self.expected_min_act[0] = (self.min_act_regularizer_batchtopk * self.expected_min_act[0]) + (
                    (1 - self.min_act_regularizer_batchtopk) * min_activation
                )

            else:
                # Do JumpReLU sparsification during inference to enable single token inference
                x = x * (x > self.expected_min_act[0])
                
            # Decode
            x = torch.matmul(codes, self.Ad.T) + self.bd

        elif self.sae_type=='MP' or self.sae_type=='mp':
            kval = self.mp_kval if inf_k is None else inf_k
            to_flatten = False
            if len(x.shape) > 2:
                to_flatten = True
                B, L, D = x.shape
                x = x.view(B*L, D)
            x = x-self.bd
            codes = torch.zeros(x.shape[0], self.Ae.shape[0], device=x.device, dtype=x.dtype)

            # Greedy selection of dictionary atoms
            for _ in range(kval):
                z = x @ self.Ae.T  # pre_codes as projection of the current residual
                val, idx = torch.max(z, dim=1)

                # add top concept to the current codes
                to_add = torch.nn.functional.one_hot(idx, num_classes=codes.shape[1]).to(dtype=x.dtype)
                to_add = to_add * val.unsqueeze(1)
                to_add = to_add * lam

                # accumulate contribution and update residual
                codes = codes + to_add
                x = x - to_add @ self.Ae

            # compute the final codes
            x = torch.matmul(codes, self.Ae) + self.bd
            if to_flatten:
                x = x.view(B, L, D)

        elif self.sae_type=='jumprelu':
            x = x-self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            x = F.relu(lam*x)
            threshold = torch.exp(self.logthreshold)
            codes = jumprelu(x, threshold, self.bandwidth)
            x = torch.matmul(codes, self.Ad.T) + self.bd
                
        elif self.sae_type=='SpaDE':
            # Encoder projection - optimized using kernel trick
            # ||x - a||^2 = ||x||^2 + ||a||^2 - 2*x^T*a
            x_norm_sq = torch.sum(x.pow(2), dim=-1, keepdim=True)  # (batch_size, 1)
            ae_norm_sq = torch.sum(self.Ae.pow(2), dim=-1, keepdim=True).T  # (1, width)
            dot_product = torch.matmul(x, self.Ae.T)  # (batch_size, width)
            x = -lam * (x_norm_sq + ae_norm_sq - 2*dot_product)

            # Sparse code: Sparsemax computation
            sm = Sparsemax(dim=-1)
            codes = sm(x)

            # Reconstruction
            x = torch.matmul(codes, self.Ad.T)

        else:
            raise ValueError('Invalid sae_type')

        if return_hidden:
            return x, codes
        else:
            return x

    @classmethod
    def from_pretrained(cls, folder_path, dtype, device, **kwargs):
        """
        Load a pretrained SAEStandard from a folder containing conf.yaml and latest_ckpt.pt

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
            'sae_type': config['sae']['sae_type'],
            'kval_topk': config['sae'].get('kval_topk'),
            'mp_kval': config['sae'].get('mp_kval'),
            'lambda_init': config['sae'].get('lambda_init'),
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