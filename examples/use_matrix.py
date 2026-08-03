#!/usr/bin/env python3
"""Load a released Matrix Market pattern and run a sample sparse matvec."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.io import mmread


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    args = parser.parse_args()

    matrix = mmread(args.matrix).tocsr().astype(np.float32)
    vector = np.ones(matrix.shape[1], dtype=np.float32)
    result = matrix @ vector
    sparsity = 1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(f"shape={matrix.shape} nnz={matrix.nnz} sparsity={sparsity:.6f}")
    print(f"SpMV result: shape={result.shape}, checksum={result.sum():.0f}")


if __name__ == "__main__":
    main()

