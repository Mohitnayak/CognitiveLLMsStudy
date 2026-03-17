#!/usr/bin/env python3
"""
Download the VisuLogic dataset from Hugging Face to a local folder.

Creates data.jsonl and images/ in the target directory so you can set
VISULOGIC_DATA_ROOT to that folder and run the VisuLogic benchmark.

Usage:
  python scripts/download_visulogic.py
  python scripts/download_visulogic.py --out-dir C:\data\VisuLogic
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "visulogic"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download VisuLogic dataset from Hugging Face.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Directory to create (data.jsonl + images/). Default: {DEFAULT_OUT}",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("This script requires huggingface_hub. Install with: pip install huggingface_hub")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "VisuLogic/VisuLogic"

    print(f"Downloading VisuLogic dataset to {out_dir} ...")
    # Download entire dataset repo (data.jsonl + images.zip) into out_dir
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=out_dir,
        allow_patterns=["data.jsonl", "images.zip"],
    )

    data_jsonl = out_dir / "data.jsonl"
    images_zip = out_dir / "images.zip"
    images_dir = out_dir / "images"

    if not data_jsonl.is_file():
        print("Error: data.jsonl was not downloaded.")
        return 1

    if images_dir.is_dir() and any(images_dir.iterdir()):
        print("images/ already present, skipping unzip.")
    elif images_zip.is_file():
        print("Unzipping images.zip ...")
        with zipfile.ZipFile(images_zip, "r") as zf:
            zf.extractall(out_dir)
        print("Done.")
    else:
        print("Warning: images.zip not found; benchmark may fail if images are required.")

    print(f"\nVisuLogic data is in: {out_dir}")
    print("Set the environment variable and run the benchmark:")
    print(f'  PowerShell: $env:VISULOGIC_DATA_ROOT = "{out_dir}"')
    print(f"  CMD:        set VISULOGIC_DATA_ROOT={out_dir}")
    print("  Then:       python scripts/run_benchmark.py --benchmark visulogic --model llava")
    return 0


if __name__ == "__main__":
    sys.exit(main())
