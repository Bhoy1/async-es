#!/usr/bin/env python3
"""Create the flat task layout expected by the official SkyRL adapter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


LINKS = {
    "container.sif": Path("environment/container.sif"),
    "container.def": Path("environment/container.def"),
    "task.json": Path("environment/task.json"),
    "test_initial_state.py": Path("environment/test_initial_state.py"),
    "test_final_state.py": Path("tests/test_final_state.py"),
    "instruction.md": Path("instruction.md"),
    "solve.sh": Path("solution/solve.sh"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("tasks/endless_terminals/data/source"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("tasks/endless_terminals/splits"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks/endless_terminals/runtime"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
    )
    return parser.parse_args()


def replace_symlink(link: Path, target: Path) -> None:
    expected = os.path.relpath(target, start=link.parent.resolve())
    if link.is_symlink() and os.readlink(link) == expected:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace nonmatching path: {link}")
    link.symlink_to(expected)


def prepare_task(source_dir: Path, output_dir: Path, task_id: str) -> None:
    source_task = source_dir / task_id
    runtime_task = output_dir / task_id
    runtime_task.mkdir(parents=True, exist_ok=True)
    for name, relative_source in LINKS.items():
        target = (source_task / relative_source).resolve()
        if not target.exists():
            raise FileNotFoundError(f"{task_id} is missing {relative_source}")
        replace_symlink(runtime_task / name, target)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()

    total = 0
    for split in args.splits:
        task_ids = [
            line.strip()
            for line in (split_dir / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        split_output = output_dir / split
        for task_id in task_ids:
            prepare_task(source_dir, split_output, task_id)
        total += len(task_ids)
        print(f"{split}: prepared {len(task_ids)} tasks", flush=True)
    print(f"prepared {total} runtime task directories under {output_dir}")


if __name__ == "__main__":
    main()
