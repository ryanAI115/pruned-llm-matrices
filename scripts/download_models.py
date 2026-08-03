#!/usr/bin/env python3
"""Download the four source checkpoints with huggingface_hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    config = json.loads((ROOT / "configs" / "models.json").read_text())
    selected = args.models or list(config)
    unknown = sorted(set(selected) - set(config))
    if unknown:
        raise SystemExit(f"Unknown model keys: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in selected:
        entry = config[key]
        destination = args.output_dir / key
        print(f"Downloading {entry['model_id']} -> {destination}", flush=True)
        snapshot_download(
            repo_id=entry["model_id"],
            local_dir=destination,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )


if __name__ == "__main__":
    main()
