#!/usr/bin/env python3
"""Download the pinned public Endless Terminals task sources."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "obiwan96/endless-terminals"
REVISION = "26ecf78458e7f756e4d06780d5fbf3dd78e91815"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks/endless_terminals/data/source"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is required; install the project with "
            "`uv pip install -e .` first"
        ) from exc
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=output_dir,
    )
    print(f"Downloaded {REPO_ID}@{REVISION} to {output_dir}")


if __name__ == "__main__":
    main()
