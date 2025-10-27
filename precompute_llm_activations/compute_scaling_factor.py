"""
For SAE traning, the expected norm of activations should match sqrt(llm_hidden_dim).
This script is computing the scaling factor.

mean(norm(acts * scaling_factor)) = sqrt(hidden_dim)
scaling_factor = sqrt(hidden_dim) / mean(norm(acts))
"""

from precompute_llm_activations.cache_utils import LocalCache
import numpy as np

cache_dir = (
    # "../../activations/precomputed_activations/precomputed_activations_gemma2/"
    "../../activations/precomputed_activations/precomputed_activations_llama3/"
)
local_cache = LocalCache(cache_dir)

for i in range(10):
    act_BLD = next(local_cache)
    # print(f"==>> act_BLD.shape: {act_BLD.shape}")

    act_ND = act_BLD.flatten(0, 1)
    N, D = act_ND.shape

    act_norm_N = act_ND.norm(p=2, dim=-1)

    scaling_factor = 1 / act_norm_N.mean()
    print(f"==>> scaling_factor: {scaling_factor.item()}")