import hydra
from omegaconf import DictConfig
import torch as th
from cache_utils import (
    precompute_activations,
    get_model_tokenizer_submodule,
    hf_dataset_to_generator,
    dtype_str_to_torch,
)


@hydra.main(version_base=None, config_path="../config/precompute", config_name="precompute_llama")
# @hydra.main(version_base=None, config_path="../config/precompute", config_name="precompute_gemma")
def main(cfg: DictConfig) -> None:
    """
    Precompute activations for SAE training.
    """
    print(f"Starting activation precomputation with config:")
    print(f"- Model: {cfg.llm_name}")
    print(f"- Layer: {cfg.layer}")
    print(f"- Dataset: {cfg.dataset_name}")
    print(f"- Total tokens: {cfg.num_total_tokens:,}")
    print(f"- Save directory: {cfg.save_dir}")

    # Convert dtype string to torch dtype
    if cfg.dtype not in dtype_str_to_torch:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}")
    dtype = dtype_str_to_torch[cfg.dtype]

    # Create environment config object for get_model_tokenizer_submodule
    class EnvConfig:
        def __init__(self, cfg):
            self.llm_name = cfg.llm_name
            self.layer = cfg.layer
            self.dtype = dtype
            self.hf_cache_dir = cfg.hf_cache_dir

    env_config = EnvConfig(cfg)

    # Load model, tokenizer, and submodule
    print("Loading model, tokenizer, and submodule...")
    model, tokenizer, submodule = get_model_tokenizer_submodule(
        env_config, do_truncate_model=cfg.do_truncate_model
    )

    # Create text generator
    print(f"Creating text generator from {cfg.dataset_name}...")
    text_generator = hf_dataset_to_generator(
        cfg.dataset_name, split=cfg.dataset_split, streaming=True
    )

    # Run precomputation
    print("Starting activation precomputation...")
    precompute_activations(
        model=model,
        submodule=submodule,
        tokenizer=tokenizer,
        text_generator=text_generator,
        num_total_tokens=cfg.num_total_tokens,
        num_tokens_per_file=cfg.num_tokens_per_file,
        num_tokens_per_batch=cfg.num_tokens_per_batch,
        context_length=cfg.context_length,
        submodule_dim=cfg.submodule_dim,
        add_special_tokens=cfg.add_special_tokens,
        save_dir=cfg.save_dir,
        device=cfg.device,
        dtype=dtype,
    )

    print("Activation precomputation completed successfully!")


if __name__ == "__main__":
    main()