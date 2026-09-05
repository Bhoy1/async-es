#!/usr/bin/env python3
"""Build fixed Endless Terminals Parquets in the official SkyRL row format."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_RUNTIME_FILES = (
    "container.sif",
    "container.def",
    "task.json",
    "test_initial_state.py",
    "test_final_state.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("tasks/endless_terminals/runtime"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("tasks/endless_terminals/splits"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks/endless_terminals/data/skyrl"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
    )
    parser.add_argument("--max-time", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assignment_string(module: ast.Module, name: str) -> str:
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        value = node.value
        strip = False
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "strip"
            and not value.args
            and not value.keywords
        ):
            value = value.func.value
            strip = True
        result = ast.literal_eval(value)
        if not isinstance(result, str):
            raise TypeError(f"{name} is not a string")
        return result.strip() if strip else result
    raise KeyError(f"could not find {name}")


def load_official_prompts(official_repo: Path) -> tuple[str, str, str]:
    prompt_source = official_repo.resolve() / "generator/sample_solutions.py"
    if not prompt_source.exists():
        raise FileNotFoundError(f"missing official prompt source: {prompt_source}")
    source_text = prompt_source.read_text()
    module = ast.parse(source_text, filename=str(prompt_source))
    system_message = assignment_string(module, "SYSTEM_MESSAGE")
    user_template = assignment_string(module, "USER_TEMPLATE")
    return system_message, user_template, hashlib.sha256(source_text.encode()).hexdigest()


def read_task_ids(split_file: Path) -> list[str]:
    task_ids = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate task IDs in {split_file}")
    return task_ids


def validate_runtime_task(task_dir: Path) -> None:
    missing = [name for name in REQUIRED_RUNTIME_FILES if not (task_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{task_dir.name} is missing: {', '.join(missing)}")


def build_row(
    task_id: str,
    split: str,
    task_dir: Path,
    system_message: str,
    user_template: str,
    max_time: int,
) -> dict[str, Any]:
    validate_runtime_task(task_dir)
    task_payload = json.loads((task_dir / "task.json").read_text())
    description = str(task_payload.get("description", "")).strip()
    if not description:
        raise ValueError(f"{task_id} has an empty task description")
    question = user_template.format(task_description=description)
    absolute_task_dir = str(task_dir.resolve())
    return {
        "data_source": "endless",
        "prompt": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        "env_class": "endless",
        "reward_spec": {
            "method": "rule",
            "ground_truth": absolute_task_dir,
        },
        "extra_info": {
            "task_id": task_id,
            "split": split,
            "task_dir": absolute_task_dir,
            "max_time": max_time,
        },
    }


def validate_round_trip(path: Path, expected_ids: list[str]) -> dict[str, Any]:
    from datasets import Dataset

    dataset = Dataset.from_parquet(str(path))
    observed_ids = [row["extra_info"]["task_id"] for row in dataset]
    if observed_ids != expected_ids:
        raise ValueError(f"task order mismatch after reloading {path}")
    required_columns = {
        "data_source",
        "prompt",
        "env_class",
        "reward_spec",
        "extra_info",
    }
    if set(dataset.column_names) != required_columns:
        raise ValueError(
            f"unexpected columns in {path}: {dataset.column_names}"
        )
    return {
        "rows": len(dataset),
        "columns": dataset.column_names,
        "features": str(dataset.features),
    }


def main() -> None:
    args = parse_args()
    if args.max_time < 1:
        raise SystemExit("--max-time must be positive")

    official_repo = args.official_repo.resolve()
    runtime_dir = args.runtime_dir.resolve()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    system_message, user_template, prompt_source_sha256 = load_official_prompts(
        official_repo
    )
    prompt_sha256 = hashlib.sha256(
        f"{system_message}\0{user_template}".encode()
    ).hexdigest()

    from datasets import Dataset

    all_ids: dict[str, list[str]] = {}
    audit_splits: dict[str, Any] = {}
    for split in args.splits:
        split_file = split_dir / f"{split}.txt"
        task_ids = read_task_ids(split_file)
        rows = [
            build_row(
                task_id,
                split,
                runtime_dir / split / task_id,
                system_message,
                user_template,
                args.max_time,
            )
            for task_id in task_ids
        ]
        output_path = output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(str(output_path))
        round_trip = validate_round_trip(output_path, task_ids)
        all_ids[split] = task_ids
        audit_splits[split] = {
            "split_file": str(split_file),
            "split_file_sha256": sha256_file(split_file),
            "rows": len(rows),
            "parquet": str(output_path),
            "parquet_sha256": sha256_file(output_path),
            "round_trip": round_trip,
        }
        print(f"{split}: wrote and validated {len(rows)} rows at {output_path}")

    overlaps: dict[str, int] = {}
    split_names = list(all_ids)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = set(all_ids[left]) & set(all_ids[right])
            overlaps[f"{left}__{right}"] = len(overlap)
            if overlap:
                raise ValueError(f"{left} and {right} overlap by {len(overlap)} tasks")

    revision = "unknown"
    git_head = official_repo / ".git/HEAD"
    if git_head.exists():
        head = git_head.read_text().strip()
        if head.startswith("ref: "):
            revision_path = official_repo / ".git" / head.removeprefix("ref: ")
            if revision_path.exists():
                revision = revision_path.read_text().strip()
        else:
            revision = head

    audit = {
        "format": "skyrl_endless_fixed_v1",
        "official_repo": str(official_repo),
        "official_revision": revision,
        "prompt_source_sha256": prompt_source_sha256,
        "prompt_sha256": prompt_sha256,
        "system_message_chars": len(system_message),
        "user_template": user_template,
        "runtime_dir": str(runtime_dir),
        "max_time": args.max_time,
        "splits": audit_splits,
        "overlaps": overlaps,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"wrote audit manifest: {manifest_path}")


if __name__ == "__main__":
    main()
