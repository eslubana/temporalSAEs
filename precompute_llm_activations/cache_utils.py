"""
Activation Cache for SAE training.
Multiple functions reused from saprmarks/dictionary-learning and tilde-research/activault
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
import torch as th
import contextlib
import os
import json
import gc
from tqdm import trange


# Dtype string to torch dtype mapping
dtype_str_to_torch = {
    "bfloat16": th.bfloat16,
    "float16": th.float16,
    "float32": th.float32,
}


######## Loading Model and Tokenizer #########


def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["text"]

    return gen()


def get_submodule(model: AutoModelForCausalLM, layer: int):
    """Gets the residual stream submodule"""
    model_name = model.name_or_path

    if model.config.architectures[0] == "GPTNeoXForCausalLM":
        return model.gpt_neox.layers[layer]
    elif (
        model.config.architectures[0] == "Qwen2ForCausalLM"
        or model.config.architectures[0] == "Gemma2ForCausalLM"
        or model.config.architectures[0] == "LlamaForCausalLM"
    ):
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


def truncate_model(model: AutoModelForCausalLM, layer: int):
    """From tilde-research/activault
    https://github.com/tilde-research/activault/blob/db6d1e4e36c2d3eb4fdce79e72be94f387eccee1/pipeline/setup.py#L74
    This provides significant memory savings by deleting all layers that aren't needed for the given layer.
    You should probably test this before using it"""
    import gc

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"Model parameters before truncation: {total_params_before:,}")

    if (
        model.config.architectures[0] == "Qwen2ForCausalLM"
        or model.config.architectures[0] == "Gemma2ForCausalLM"
    ):
        removed_layers = model.model.layers[layer + 1 :]

        model.model.layers = model.model.layers[: layer + 1]

        del removed_layers
        del model.lm_head

        model.lm_head = th.nn.Identity()

    elif model.config.architectures[0] == "GPTNeoXForCausalLM":
        removed_layers = model.gpt_neox.layers[layer + 1 :]

        model.gpt_neox.layers = model.gpt_neox.layers[: layer + 1]

        del removed_layers
        del model.embed_out

        model.embed_out = th.nn.Identity()

    else:
        raise ValueError(f"Please add truncation for model {model.name_or_path}")

    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"Model parameters after truncation: {total_params_after:,}")

    gc.collect()
    th.cuda.empty_cache()

    return model


def get_model_tokenizer_submodule(env_config, do_truncate_model=False):

    model = AutoModelForCausalLM.from_pretrained(
        env_config.llm_name,
        device_map="auto",
        torch_dtype=env_config.dtype,
        cache_dir=env_config.hf_cache_dir,
    )

    if do_truncate_model:
        model = truncate_model(model, env_config.layer)

    tokenizer = AutoTokenizer.from_pretrained(env_config.llm_name)
    submodule = get_submodule(model, env_config.layer)
    return model, tokenizer, submodule


######## Collecting Activations #########


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: th.nn.Module,
    inputs_BL: dict[str, th.Tensor],
    use_no_grad: bool = True,
) -> th.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `t.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = th.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    if activations_BLD is None:
        # This should ideally not happen if the hook worked and EarlyStopException was raised,
        # but handle it just in case.
        raise RuntimeError("Failed to collect activations. The hook might not have run correctly.")

    return activations_BLD


def tokenized_batch(
    tokenizer,
    text_generator,
    num_tokens_per_batch,
    context_length,
    add_special_tokens,
):
    """
    Return a batch of tokenized inputs.
    """
    num_seq_per_batch = num_tokens_per_batch // context_length

    batch = []
    while len(batch) < num_seq_per_batch:
        try:
            sample_text = next(text_generator)
        except StopIteration:
            raise StopIteration("End of data stream reached")

        sample_tok_L = tokenizer(
            sample_text,
            return_tensors="pt",
            max_length=context_length,
            truncation=True,
            padding=False,
            add_special_tokens=add_special_tokens,
        )

        # Only select texts longer or equal than ctx_len
        if sample_tok_L.input_ids.shape[1] < context_length:
            continue
        else:
            batch.append(sample_tok_L)
        # TODO make sure that no padding occurs across the whole repo!

    ids_BL = th.cat([s.input_ids for s in batch], dim=0)
    mask_BL = th.cat([s.attention_mask for s in batch], dim=0)
    encoding_BL = BatchEncoding(
        {
            "input_ids": ids_BL,
            "attention_mask": mask_BL,  # Mask is expected to be all true
        }
    )

    return encoding_BL


def tokenized_batch_generator(
    dataset_name,
    split,
    tokenizer,
    num_tokens_per_batch,
    ctx_len,
    add_special_tokens=True,
    max_batches=10000,
):
    text_generator = hf_dataset_to_generator(dataset_name, split, streaming=True)

    for _ in range(max_batches):
        yield tokenized_batch(
            tokenizer,
            text_generator,
            num_tokens_per_batch,
            ctx_len,
            add_special_tokens=add_special_tokens,
        )


def generate_metadata(save_dir, num_files):
    """Generate metadata.json file compatible with S3RCache."""
    first_states_path = os.path.join(save_dir, f"states_{0:05d}_of_{num_files:05d}.pkl")
    first_input_ids_path = os.path.join(save_dir, f"input_ids_{0:05d}_of_{num_files:05d}.pkl")

    # Load first files to get tensor shapes and dtype
    with open(first_states_path, "rb") as f:
        first_states = th.load(f, weights_only=False)
    with open(first_input_ids_path, "rb") as f:
        first_input_ids = th.load(f, weights_only=False)

    states_shape = list(first_states.shape)
    input_ids_shape = list(first_input_ids.shape)
    dtype_str = str(first_states.dtype)

    # Get file sizes in bytes
    states_bytes_per_file = os.path.getsize(first_states_path)
    input_ids_bytes_per_file = os.path.getsize(first_input_ids_path)

    metadata = {
        "shape": states_shape,
        "input_ids_shape": input_ids_shape,
        "dtype": dtype_str,
        "states_bytes_per_file": states_bytes_per_file,
        "input_ids_bytes_per_file": input_ids_bytes_per_file,
    }

    # Save metadata.json
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")
    return metadata


def precompute_activations(
    model,
    submodule,
    tokenizer,
    text_generator,
    num_total_tokens,
    num_tokens_per_file,
    num_tokens_per_batch,
    context_length,
    submodule_dim,
    add_special_tokens,
    save_dir,
    device,
    dtype,
) -> None:

    os.makedirs(save_dir, exist_ok=True)

    assert (
        num_total_tokens % num_tokens_per_file == 0
    ), "Num_total_tokens must be a multiple of tokens_per_file"
    assert (
        num_tokens_per_file % num_tokens_per_batch == 0
    ), "Num_tokens_per_file must be a multiple of num_tokens_per_batch"

    num_files = num_total_tokens // num_tokens_per_file
    num_batches_per_file = num_tokens_per_file // num_tokens_per_batch
    num_seq_per_batch = num_tokens_per_batch // context_length

    for file_idx in range(num_files):
        file_acts_nBLD = th.zeros(
            num_batches_per_file,
            num_seq_per_batch,
            context_length,
            submodule_dim,
            device=device,
            dtype=dtype,
        )
        file_inputs_nBL = th.zeros(
            num_batches_per_file, num_seq_per_batch, context_length, device=device, dtype=dtype
        )
        for batch_idx in trange(
            num_batches_per_file, desc=f"Processing file {file_idx}/{num_files}"
        ):
            # Collect input token batch
            encoding_BL = tokenized_batch(
                tokenizer=tokenizer,
                text_generator=text_generator,
                num_tokens_per_batch=num_tokens_per_batch,
                context_length=context_length,
                add_special_tokens=add_special_tokens,
            )
            file_inputs_nBL[batch_idx] = encoding_BL.input_ids

            # Collect LLM activations
            encoding_BL = encoding_BL.to(device)
            acts_BLD = collect_activations(model, submodule, encoding_BL)
            file_acts_nBLD[batch_idx] = acts_BLD

        # Save states and input_ids as separate files
        states_name = f"states_{file_idx:05d}_of_{num_files:05d}.pkl"
        input_ids_name = f"input_ids_{file_idx:05d}_of_{num_files:05d}.pkl"

        states_path = os.path.join(save_dir, states_name)
        input_ids_path = os.path.join(save_dir, input_ids_name)

        # Save states file
        with open(states_path, "wb") as f:
            th.save(file_acts_nBLD.flatten(0, 1).cpu(), f)

        # Save input_ids file
        with open(input_ids_path, "wb") as f:
            th.save(file_inputs_nBL.flatten(0, 1).cpu(), f)

        del file_inputs_nBL, file_acts_nBLD
        gc.collect()
        th.cuda.empty_cache()

    # Generate metadata.json after all files are saved
    generate_metadata(save_dir, num_files)

    print(f"LLM activations successfully precomputed!")


######### Activation Buffer for loading precomputed acts during training ########


class LocalCache:
    """Lightweight cache that loads precomputed activation files sequentially. Only loads states for faster performance."""

    def __init__(self, save_dir):
        self.save_dir = save_dir

        # Load metadata
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        B, L, D = self.metadata["shape"]
        self.num_seq_per_file = B
        self.context_length = L
        self.hidden_dim = D

        # Find all states files (only load states for performance)
        self.states_file_paths = []

        for filename in os.listdir(save_dir):
            if filename.startswith("states_") and filename.endswith(".pkl"):
                self.states_file_paths.append(os.path.join(save_dir, filename))

        self.states_file_paths.sort()  # Ensure consistent order
        self.current_file_idx = 0

    def __iter__(self):
        self.current_file_idx = 0
        return self

    def __next__(self):
        if self.current_file_idx >= len(self.states_file_paths):
            raise StopIteration

        states_path = self.states_file_paths[self.current_file_idx]
        self.current_file_idx += 1

        with open(states_path, "rb") as f:
            states = th.load(f, weights_only=False)

        return states


class ActivationPrecomputedBuffer:
    def __init__(
        self,
        cache_dir: str,
        context_length: int,
        num_seq_per_batch: int,  # num total tokens per batch = num_sequences * context_length (not num_sequences!)
    ):
        self.local_cache = LocalCache(cache_dir)

        self.context_length = context_length
        assert (
            context_length == self.local_cache.context_length
        ), f"Context Length mismatch! Activation Buffer: {context_length}, Local Cache: {self.local_cache}"

        self.num_seq_per_batch = num_seq_per_batch

        self.buffer_BLD: th.tensor = (
            None,
        )  # Activation buffer of shape B (num sequences), L (context length), D (model hidden dim)
        self.num_read: int = None

        self.refresh()

    def refresh(self):
        # Clear cache
        self.buffer_BLD = None
        th.cuda.empty_cache()
        gc.collect()

        self.buffer_BLD = next(self.local_cache)
        self.num_read = 0

    def __next__(self):
        if self.num_read > self.buffer_BLD.shape[0] - self.num_seq_per_batch:
            self.refresh()

        next_batch_bLD = self.buffer_BLD[self.num_read : self.num_read + self.num_seq_per_batch]
        self.num_read += self.num_seq_per_batch

        return next_batch_bLD

    def __iter__(self):
        return self

    def cleanup(self):
        """Clean up memory and resources."""
        self.buffer_BLD = None
        th.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    buffer = ActivationPrecomputedBuffer(
        cache_dir="./asdasdasdasdasdasdsadsadasdas",
        context_length=500,
        num_seq_per_batch=2,
    )
    acts = next(buffer)
    print(f"==>> acts.shape: {acts.shape}")