#!/usr/bin/env python3
"""Download, checksum, and extract the split data release."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import zstandard


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_drive_url(url: str) -> str:
    """Turn a public Drive file URL into a direct-download URL."""
    parsed = urlparse(url)
    if parsed.netloc not in {"drive.google.com", "www.drive.google.com"}:
        return url
    parts = parsed.path.split("/")
    if "d" in parts:
        file_id = parts[parts.index("d") + 1]
        return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    return url


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(normalize_drive_url(url), stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def safe_extract_zstd(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for member in tar:
                    target = (output_dir / member.name).resolve()
                    if output_dir.resolve() not in target.parents and target != output_dir.resolve():
                        raise ValueError(f"Unsafe archive member: {member.name}")
                    if not (member.isdir() or member.isreg()):
                        raise ValueError(f"Unsupported archive member: {member.name}")
                    tar.extract(member, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    release = json.loads((ROOT / "data" / "archives.json").read_text())
    archives = release["archives"]
    selected = set(args.models or [item["model_key"] for item in archives])
    unknown = sorted(selected - {item["model_key"] for item in archives})
    if unknown:
        raise SystemExit(f"Unknown model keys: {', '.join(unknown)}")

    archive_dir = args.output_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for item in archives:
        if item["model_key"] not in selected:
            continue
        archive = archive_dir / item["file_name"]
        if not archive.exists() or sha256(archive) != item["sha256"]:
            part_paths = []
            for part in item["parts"]:
                part_path = archive_dir / part["file_name"]
                if not part_path.exists() or sha256(part_path) != part["sha256"]:
                    print(
                        f"Downloading {part['file_name']} ({part['size_bytes']} bytes)"
                    )
                    download(part["url"], part_path)
                observed_part = sha256(part_path)
                if observed_part != part["sha256"]:
                    raise ValueError(
                        f"Checksum mismatch for {part_path}: {observed_part}"
                    )
                part_paths.append(part_path)
            print(f"Joining {len(part_paths)} parts -> {archive.name}")
            partial_archive = archive.with_suffix(archive.suffix + ".part")
            with partial_archive.open("wb") as destination:
                for part_path in part_paths:
                    with part_path.open("rb") as source:
                        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            destination.write(chunk)
            partial_archive.replace(archive)
        observed = sha256(archive)
        if observed != item["sha256"]:
            raise ValueError(f"Checksum mismatch for {archive}: {observed}")
        print(f"Extracting {archive.name}")
        safe_extract_zstd(archive, args.output_dir)
        if not args.keep_archives:
            archive.unlink()
            for part in item["parts"]:
                (archive_dir / part["file_name"]).unlink(missing_ok=True)

    if not args.keep_archives:
        try:
            archive_dir.rmdir()
        except OSError:
            pass
    print(f"Dataset ready under {args.output_dir}")


if __name__ == "__main__":
    main()
