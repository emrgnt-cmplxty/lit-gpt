import sys
import warnings
from pathlib import Path
from typing import Literal, Optional

import torch
from transformers import LlamaTokenizer
from dotenv import load_dotenv
import os
from lit_gpt.adapter import GPT, Config

load_dotenv()

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    lazy_load,
    quantization,
)

from torch.nn import functional as F


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    max_seq_length: int = None,
    eos_id: int = 2,
) -> torch.Tensor:
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
        # forward the model to get the logits for the index in the sequence
        logits = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next == eos_id:
            return idx  # [:input_pos]  # include the EOS token

        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# def generate(
#     model: torch.nn.Module,
#     idx: torch.Tensor,
#     max_returned_tokens: int,
#     max_seq_length: int,
#     *,
#     temperature: float = 1.0,
#     top_k: Optional[int] = None,
#     eos_id: Optional[int] = None,
# ) -> torch.Tensor:
#     """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

#     The implementation of this function is modified from A. Karpathy's nanoGPT.

#     Args:
#         model: The model to use.
#         idx: Tensor of shape (T) with indices of the prompt sequence.
#         max_returned_tokens: The maximum number of tokens to return (given plus generated).
#         max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
#         temperature: Scales the predicted logits by 1 / temperature.
#         top_k: If specified, only sample among the tokens with the k highest probabilities.
#         eos_id: If specified, stop generating any more token once the <eos> token is triggered.
#     """
#     T = idx.size(0)
#     assert max_returned_tokens > T
#     device, dtype = idx.device, idx.dtype
#     # create an empty tensor of the expected final shape and fill in the current tokens
#     empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
#     empty[:T] = idx
#     idx = empty
#     input_pos = torch.arange(0, T, device=device)

#     # generate up to a fixed number of tokens
#     for _ in range(max_returned_tokens - T):
#         x = idx.index_select(0, input_pos).view(1, -1)

#         # forward
#         logits = model(x)  # , max_seq_length, input_pos)
#         logits = logits[0, -1] / temperature

#         # optionally crop the logits to only the top k options
#         if top_k is not None:
#             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#             logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

#         probs = torch.nn.functional.softmax(logits, dim=-1)
#         idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

#         # advance
#         input_pos = input_pos[-1:] + 1

#         # concatenate the new generation
#         idx = idx.index_copy(0, input_pos, idx_next)

#         # if <eos> token is triggered, return the output (stop generation)
#         if idx_next == eos_id:
#             return idx[:input_pos]  # include the EOS token

#     return idx


def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[
        Literal[
            "bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"
        ]
    ] = None,
    strategy: str = "auto",
    device: str = "cpu",
    tokenizer: str = "meta-llama/Llama-2-7b",
    precision: Optional[str] = None,
    model_name: str = "SmolKat_10M",
    eos_id: int = -1,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    ckpt = torch.load(checkpoint_dir, map_location=device)
    unwanted_prefix = "module._orig_mod."
    state_dict = ckpt["state_dict"]
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    unwanted_prefix = "module."
    state_dict = ckpt["state_dict"]
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    print("ckpt.keys() = ", ckpt.keys())

    config = Config.from_name(name=model_name)
    print("config = ", config)
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    hf_token = os.getenv("HF_TOKEN", None)
    enc = LlamaTokenizer.from_pretrained(tokenizer, token=hf_token)

    encoded = torch.Tensor(enc(prompt)["input_ids"]).long().to(device)[None, ...]
    max_returned_tokens = 64
    y = generate(
        model,
        encoded,
        max_returned_tokens,
        max_seq_length=max_returned_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )
    z = enc.decode(y[0].tolist())
    print("z = ", z)

    # prompt_length = encoded.size(0)
    # max_returned_tokens = prompt_length + max_new_tokens
    # # assert max_returned_tokens <= model.config.block_size, (
    # #     max_returned_tokens,
    # #     model.config.block_size,
    # # )  # maximum rope cache length

    # L.seed_everything(1234)
    # for i in range(num_samples):
    #     t0 = time.perf_counter()
    #     y = generate(
    #         model,
    #         encoded,
    #         max_returned_tokens,
    #         max_seq_length=max_returned_tokens,
    #         temperature=temperature,
    #         top_k=top_k,
    #     )
    #     t = time.perf_counter() - t0

    #     model.reset_cache()
    #     fabric.print(tokenizer.decode(y))
    #     tokens_generated = y.size(0) - prompt_length
    #     fabric.print(
    #         f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
    #     )
    # if fabric.device.type == "cuda":
    #     fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)
