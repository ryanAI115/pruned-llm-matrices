#!/usr/bin/env python3
"""Verify released MTX coverage and optionally load every sparsity pattern."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[1]


def digest_from_sparse(matrix) -> str:
    coo = matrix.tocoo()
    mask = np.zeros(coo.shape, dtype=np.bool_)
    mask[coo.row, coo.col] = True
    return hashlib.sha256(np.packbits(mask, axis=None).tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["matrix"]: row for row in rows}
    actual = {path.stem: path for path in args.matrix_dir.rglob("*.mtx")}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        print(f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("first missing:", *missing[:10], sep="\n  ")
        if unexpected:
            print("first unexpected:", *unexpected[:10], sep="\n  ")
        raise SystemExit(1)

    if args.deep:
        for index, (name, row) in enumerate(expected.items(), start=1):
            matrix = mmread(actual[name]).tocsr()
            expected_shape = (int(row["rows"]), int(row["cols"]))
            if matrix.shape != expected_shape:
                raise ValueError(f"{name}: shape {matrix.shape} != {expected_shape}")
            if matrix.nnz != int(row["nnz"]):
                raise ValueError(f"{name}: nnz {matrix.nnz} != {row['nnz']}")
            digest = digest_from_sparse(matrix)
            if digest != row["mask_sha256_16"]:
                raise ValueError(
                    f"{name}: mask digest {digest} != {row['mask_sha256_16']}"
                )
            if index % 50 == 0 or index == len(expected):
                print(f"deep-verified {index}/{len(expected)}", flush=True)

    print(f"OK: {len(expected)} matrices match {args.manifest}")


if __name__ == "__main__":
    main()

