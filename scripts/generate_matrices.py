#!/usr/bin/env python3
"""Extract all q/o projection weights, prune them, and write pattern MTX files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmwrite


METHODS = tuple(
    (f"{score}_s{sparsity}", score, sparsity / 100.0)
    for sparsity in (70, 80, 90)
    for score in ("global_mag", "wanda")
)


def exact_mask(score: np.ndarray, sparsity: float) -> np.ndarray:
    """Keep exactly round(numel * (1-sparsity)) largest scoring entries."""
    flat = score.reshape(-1)
    keep = max(1, round(flat.size * (1.0 - sparsity)))
    chosen = np.argpartition(flat, flat.size - keep)[flat.size - keep :]
    mask = np.zeros(flat.size, dtype=np.bool_)
    mask[chosen] = True
    return mask.reshape(score.shape)


def mask_digest(mask: np.ndarray) -> str:
    packed = np.packbits(mask, axis=None).tobytes()
    return hashlib.sha256(packed).hexdigest()[:16]


def write_pattern(path: Path, mask: np.ndarray) -> None:
    matrix = sparse.coo_matrix(mask.astype(np.int8, copy=False))
    mmwrite(path, matrix, field="pattern", symmetry="general", precision=None)


def normalized_module(name: str) -> str | None:
    if name.endswith("self_attn.q_proj"):
        return "q_proj"
    if name.endswith(("self_attn.o_proj", "self_attn.out_proj")):
        return "o_proj"
    return None


def layer_number(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--activation-scales", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    if (args.output_dir / "manifest.csv").exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {args.output_dir}")
    matrix_dir = args.output_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    scales = np.load(args.activation_scales)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=not args.allow_download,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    selected = []
    available_layers: set[int] = set()
    for name, module in model.named_modules():
        module_label = normalized_module(name)
        layer = layer_number(name)
        if module_label and layer is not None and hasattr(module, "weight"):
            available_layers.add(layer)
            selected.append((layer, module_label, name, module))

    requested_layers = set(args.layers) if args.layers else available_layers
    if not requested_layers <= available_layers:
        missing = sorted(requested_layers - available_layers)
        raise RuntimeError(f"Requested layers not found: {missing}")
    selected = [item for item in selected if item[0] in requested_layers]
    selected.sort(key=lambda item: (item[0], item[1]))
    expected = len(requested_layers) * 2
    if len(selected) != expected:
        names = [item[2] for item in selected]
        raise RuntimeError(f"Expected {expected} q/o modules, found {len(selected)}: {names}")

    manifest = []
    for layer, module_label, tensor_prefix, module in selected:
        weight = module.weight.detach().float().cpu().numpy()
        abs_weight = np.abs(weight)
        tensor = f"{tensor_prefix}.weight"
        if tensor not in scales:
            raise KeyError(f"Activation scales do not contain {tensor}")
        activation = np.asarray(scales[tensor], dtype=np.float32)
        if activation.shape != (weight.shape[1],):
            raise ValueError(f"Activation scale mismatch for {tensor}: {activation.shape}")
        wanda_score = abs_weight * activation[None, :]

        for method, score_name, target in METHODS:
            score = abs_weight if score_name == "global_mag" else wanda_score
            mask = exact_mask(score, target)
            matrix_name = f"{args.model_key}_l{layer:02d}_{module_label}_{method}"
            write_pattern(matrix_dir / f"{matrix_name}.mtx", mask)
            nnz = int(mask.sum())
            manifest.append(
                {
                    "matrix": matrix_name,
                    "source_model": args.model_label,
                    "model_key": args.model_key,
                    "source_layer": tensor,
                    "layer_group": f"{args.model_key}_block_{layer:02d}",
                    "transformer_block": layer,
                    "module": module_label,
                    "shape_family": f"{weight.shape[0]}x{weight.shape[1]}",
                    "method": method,
                    "pruning_family": (
                        "unstructured"
                        if score_name == "global_mag"
                        else "activation_aware_unstructured"
                    ),
                    "target_sparsity": target,
                    "pruning_block_size": "",
                    "rows": weight.shape[0],
                    "cols": weight.shape[1],
                    "nnz": nnz,
                    "realized_sparsity": 1.0 - nnz / mask.size,
                    "mask_sha256_16": mask_digest(mask),
                    "activation_calibration": (
                        args.activation_scales.name if score_name == "wanda" else ""
                    ),
                }
            )
            print(f"{matrix_name}: shape={weight.shape} nnz={nnz}", flush=True)
        del weight, abs_weight, wanda_score

    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    metadata = {
        "model": args.model,
        "model_key": args.model_key,
        "layers": sorted(requested_layers),
        "modules": ["q_proj", "o_proj"],
        "methods": [method[0] for method in METHODS],
        "methods_per_source_matrix": len(METHODS),
        "matrices": len(manifest),
        "wanda_score": "abs(weight) * sqrt(mean(input_activation^2))",
        "scope": "all-layer kernel-selection patterns; no perplexity claim",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote {len(manifest)} matrices for {args.model_label}")


if __name__ == "__main__":
    main()
