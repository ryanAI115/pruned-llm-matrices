#!/usr/bin/env python3
"""Collect deterministic input-channel RMS activations for q/o projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROMPTS = [
    "Explain why sparse matrix vector multiplication is memory bound.",
    "Write a Python function that computes the greatest common divisor.",
    "Summarize the causes of the French Revolution in five sentences.",
    "A train travels 120 kilometers in 80 minutes. Compute its average speed.",
    "Compare structured pruning, unstructured pruning, and N:M sparsity.",
    "Translate into Chinese: Efficient inference requires careful kernel selection.",
    "Describe the difference between autoregressive decoding and prompt prefill.",
    "Prove that the square root of two is irrational.",
    "Design an experiment with a held-out test group and multiple random seeds.",
    "What are the tradeoffs between CSR, COO, ELL, and blocked sparse formats?",
    "Implement binary search and state its time complexity.",
    "Explain attention heads and grouped-query attention to a systems researcher.",
    "List three reasons a classifier's accuracy may not predict runtime speedup.",
    "Given matrix A and vector x, derive the dimensions of y = A x.",
    "Discuss how pruning granularity changes accelerator utilization.",
    "Create a concise benchmark report with assumptions, results, and limitations.",
]


def wanted_projection(name: str) -> bool:
    return name.endswith(
        ("self_attn.q_proj", "self_attn.o_proj", "self_attn.out_proj")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=local_only,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device).eval()

    sums: dict[str, object] = {}
    counts: dict[str, int] = {}
    handles = []

    def hook_for(name: str):
        def hook(_module, inputs):
            activation = inputs[0].detach().float()
            reduce_dims = tuple(range(activation.ndim - 1))
            channel_sum = activation.square().sum(dim=reduce_dims).cpu()
            sums[name] = sums.get(name, torch.zeros_like(channel_sum)) + channel_sum
            counts[name] = counts.get(name, 0) + (
                activation.numel() // activation.shape[-1]
            )

        return hook

    for name, module in model.named_modules():
        if wanted_projection(name):
            handles.append(module.register_forward_pre_hook(hook_for(name)))

    encoded = tokenizer(
        PROMPTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    ).to(args.device)
    with torch.inference_mode():
        model(**encoded, use_cache=False)
    for handle in handles:
        handle.remove()

    arrays = {
        f"{name}.weight": torch.sqrt(sums[name] / counts[name])
        .numpy()
        .astype(np.float32)
        for name in sorted(sums)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "model": args.model,
        "model_key": args.model_key,
        "prompts": PROMPTS,
        "max_length": args.max_length,
        "modules": len(arrays),
        "tokens_including_padding": counts,
        "metric": "sqrt(mean(input_activation^2)) per input channel",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote activation scales for {len(arrays)} modules to {args.output}")


if __name__ == "__main__":
    main()
