#!/usr/bin/env python3
"""Build reproducible Endless Terminals inventory and split manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


SOURCE_REPO = "obiwan96/endless-terminals"
SOURCE_REVISION = "26ecf78458e7f756e4d06780d5fbf3dd78e91815"
SUMMARY_FILES = {
    "o3": "o3_summary.json",
    "llama_3b": "meta-llama_Llama-3.2-3B-Instruct_summary.json",
}
COMMAND_RE = re.compile(r"<command>(.*?)</command>", re.IGNORECASE | re.DOTALL)
URL_RE = re.compile(r"https?://[^\s`<>\"]+", re.IGNORECASE)
NETWORK_RE = re.compile(
    r"\b(?:probe|fetch|download|curl|wget|request|resolve|ping|connect|"
    r"public endpoint|remote (?:host|server)|api endpoint)\b",
    re.IGNORECASE,
)
NO_NETWORK_RE = re.compile(
    r"\b(?:do not|don't|must not|no need to|without)\b.{0,50}\b(?:internet|network)\b",
    re.IGNORECASE | re.DOTALL,
)
DOMAIN_PATTERNS = {
    "network_web": r"\b(?:https?|curl|wget|dns|hostname|endpoint|webserver|api)\b",
    "database": r"\b(?:sql|sqlite|database|postgres|mysql|query|table)\b",
    "archives": r"\b(?:archive|tar|gzip|bzip|xz|zip|compress|extract)\b",
    "logs_audit": r"\b(?:logs?|audit|syslog|journal|diagnostic)\b",
    "structured_data": r"\b(?:csv|json|yaml|xml|dataset|spreadsheet)\b",
    "text_processing": r"\b(?:regex|regular expression|grep|sed|awk|text file|lines?)\b",
    "security_permissions": r"\b(?:chmod|permission|security|harden|ssh|certificate|openssl)\b",
    "build_packages": r"\b(?:compile|build|makefile|package|dependency|install)\b",
    "scripting": r"\b(?:bash|shell script|python script|script)\b",
    "file_operations": r"\b(?:files?|director(?:y|ies)|copy|move|rename|symlink)\b",
}
DOMAIN_WEIGHTS = {
    "network_web": 2.5,
    "database": 3.0,
    "archives": 3.0,
    "logs_audit": 2.0,
    "structured_data": 2.0,
    "text_processing": 1.25,
    "security_permissions": 2.5,
    "build_packages": 2.5,
    "scripting": 1.5,
    "file_operations": 0.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("tasks/endless_terminals/data/source"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks/endless_terminals/manifests"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("tasks/endless_terminals/splits"),
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=100,
        help="Number of stable tasks reserved for model selection.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=300,
        help="Number of stable tasks reserved for final evaluation.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="Held-out fraction for the separate context-rich split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("tasks/endless_terminals/policies/task_filters.toml"),
    )
    return parser.parse_args()


def median(values: list[int | float]) -> float | None:
    return float(statistics.median(values)) if values else None


def command_from_message(message: dict[str, Any]) -> str | None:
    if message.get("role") != "assistant":
        return None
    content = str(message.get("content", ""))
    match = COMMAND_RE.search(content)
    return match.group(1).strip() if match else None


def trajectory_metrics(result: dict[str, Any]) -> dict[str, Any]:
    transcript = result.get("transcript") or []
    commands = [command for message in transcript if (command := command_from_message(message))]
    assistant_turns = sum(message.get("role") == "assistant" for message in transcript)
    observation_chars = sum(
        len(str(message.get("content", "")))
        for message in transcript[2:]
        if message.get("role") == "user"
    )
    repeated_commands = sum(a == b for a, b in zip(commands, commands[1:]))
    command_pairs = max(0, len(commands) - 1)
    error_observations = sum(
        "command failed" in str(message.get("content", "")).lower()
        for message in transcript[2:]
        if message.get("role") == "user"
    )
    return {
        "success": bool(result.get("success")),
        "assistant_turns": assistant_turns,
        "command_turns": len(commands),
        "transcript_chars": sum(len(str(message.get("content", ""))) for message in transcript),
        "history_chars": sum(
            len(str(message.get("content", ""))) for message in transcript[2:]
        ),
        "observation_chars": observation_chars,
        "repeat_rate": repeated_commands / command_pairs if command_pairs else 0.0,
        "error_observations": error_observations,
    }


def summarize_trajectories(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text())
    runs = [trajectory_metrics(result) for result in payload.get("results", [])]
    successful = [run for run in runs if run["success"]]
    failed = [run for run in runs if not run["success"]]
    num_runs = int(payload.get("num_runs", len(runs)))
    num_success = int(payload.get("num_success", len(successful)))
    return {
        "available": True,
        "num_runs": num_runs,
        "num_success": num_success,
        "empirical_pass_1": num_success / num_runs if num_runs else None,
        "success_command_turns_median": median([run["command_turns"] for run in successful]),
        "success_command_turns_max": max((run["command_turns"] for run in successful), default=None),
        "success_assistant_turns_median": median([run["assistant_turns"] for run in successful]),
        "success_transcript_chars_median": median([run["transcript_chars"] for run in successful]),
        "success_history_chars_median": median([run["history_chars"] for run in successful]),
        "success_observation_chars_median": median([run["observation_chars"] for run in successful]),
        "failed_repeat_rate_mean": (
            statistics.fmean(run["repeat_rate"] for run in failed) if failed else None
        ),
        "failed_error_observations_mean": (
            statistics.fmean(run["error_observations"] for run in failed) if failed else None
        ),
    }


def context_tier(o3: dict[str, Any], llama: dict[str, Any]) -> str:
    preferred = o3 if o3.get("num_success", 0) else llama
    turns = preferred.get("success_command_turns_median") or 0
    chars = preferred.get("success_history_chars_median") or 0
    if turns >= 8 or chars >= 8_000:
        return "long"
    if turns >= 4 or chars >= 2_000:
        return "medium"
    if preferred.get("num_success", 0):
        return "short"
    return "unknown"


def stable_hash(seed: int, task_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def derive_domains(instruction: str) -> list[str]:
    scores = {
        domain: len(re.findall(pattern, instruction, re.IGNORECASE))
        * DOMAIN_WEIGHTS[domain]
        for domain, pattern in DOMAIN_PATTERNS.items()
    }
    domains = [domain for domain, score in scores.items() if score > 0]
    return sorted(domains, key=lambda domain: (-scores[domain], domain)) or ["other"]


def external_urls(instruction: str) -> list[str]:
    local_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    urls = [match.group(0).rstrip(".,;:)]}") for match in URL_RE.finditer(instruction)]
    return [url for url in urls if (urlsplit(url).hostname or "").lower() not in local_hosts]


def build_row(task_dir: Path, exclusions: dict[str, str]) -> dict[str, Any]:
    instruction_path = task_dir / "instruction.md"
    config_path = task_dir / "task.toml"
    task_json_path = task_dir / "environment" / "task.json"
    dockerfile_path = task_dir / "environment" / "Dockerfile"
    container_path = task_dir / "environment" / "container.def"
    instruction = instruction_path.read_text() if instruction_path.exists() else ""
    config = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
    task_json = json.loads(task_json_path.read_text()) if task_json_path.exists() else {}
    metadata = config.get("metadata", {})
    environment = config.get("environment", {})
    verifier = config.get("verifier", {})
    agent = config.get("agent", {})
    dockerfile = dockerfile_path.read_text() if dockerfile_path.exists() else ""
    o3 = summarize_trajectories(task_dir / "solution" / SUMMARY_FILES["o3"])
    llama = summarize_trajectories(task_dir / "solution" / SUMMARY_FILES["llama_3b"])
    required = [instruction_path, config_path, task_json_path]
    structure_complete = all(path.exists() for path in required) and (
        dockerfile_path.exists() or container_path.exists()
    )
    public_urls = external_urls(instruction)
    external_url = bool(public_urls)
    network_review = bool(URL_RE.search(instruction) or NETWORK_RE.search(instruction))
    runtime_network = bool(
        external_url
        and NETWORK_RE.search(instruction)
        and not NO_NETWORK_RE.search(instruction)
    )
    build_network = bool(re.search(r"\b(?:apt-get|pip|npm|git clone|curl|wget)\b", dockerfile))
    learnable = bool(o3.get("num_success", 0))
    tier = context_tier(o3, llama)
    derived_domains = derive_domains(instruction)
    excluded_reasons = [exclusions[task_dir.name]] if task_dir.name in exclusions else []
    stable_candidate = structure_complete and not excluded_reasons
    return {
        "task_id": task_dir.name,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "instruction_chars": len(instruction),
        "instruction_words": len(instruction.split()),
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "description_matches_instruction": task_json.get("description") == instruction.strip(),
        "difficulty": metadata.get("difficulty", "unknown"),
        "category": metadata.get("category", "unknown"),
        "primary_domain": derived_domains[0],
        "derived_domains": derived_domains,
        "tags": metadata.get("tags", []),
        "agent_timeout_sec": agent.get("timeout_sec"),
        "verifier_timeout_sec": verifier.get("timeout_sec"),
        "build_timeout_sec": environment.get("build_timeout_sec"),
        "cpus": environment.get("cpus"),
        "memory_mb": environment.get("memory_mb"),
        "storage_mb": environment.get("storage_mb"),
        "structure_complete": structure_complete,
        "requires_runtime_network": runtime_network,
        "network_review": network_review,
        "external_urls": public_urls,
        "uses_build_network": build_network,
        "stable_candidate": stable_candidate,
        "excluded_reasons": excluded_reasons,
        "learnable_by_o3": learnable,
        "context_tier": tier,
        "o3": o3,
        "llama_3b": llama,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_ids(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(f"{row['task_id']}\n" for row in rows))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "task_id",
        "difficulty",
        "category",
        "primary_domain",
        "context_tier",
        "stable_candidate",
        "learnable_by_o3",
        "requires_runtime_network",
        "instruction_words",
        "agent_timeout_sec",
        "o3_num_success",
        "o3_empirical_pass_1",
        "o3_success_command_turns_median",
        "o3_success_transcript_chars_median",
        "o3_success_history_chars_median",
        "llama_3b_num_success",
        "llama_3b_empirical_pass_1",
        "llama_3b_success_command_turns_median",
        "llama_3b_success_transcript_chars_median",
        "llama_3b_success_history_chars_median",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in columns}
            for model in ("o3", "llama_3b"):
                for metric in (
                    "num_success",
                    "empirical_pass_1",
                    "success_command_turns_median",
                    "success_transcript_chars_median",
                    "success_history_chars_median",
                ):
                    flat[f"{model}_{metric}"] = row[model].get(metric)
            writer.writerow(flat)


def stratified_split(
    rows: list[dict[str, Any]], eval_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["primary_domain"], row["difficulty"], row["context_tier"])
        strata[key].append(row)
    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for values in strata.values():
        ordered = sorted(values, key=lambda row: stable_hash(seed, row["task_id"]))
        eval_count = round(len(ordered) * eval_fraction)
        if len(ordered) > 1:
            eval_count = min(max(eval_count, 1), len(ordered) - 1)
        evaluation.extend(ordered[:eval_count])
        train.extend(ordered[eval_count:])
    return sorted(train, key=lambda row: row["task_id"]), sorted(
        evaluation, key=lambda row: row["task_id"]
    )


def fixed_stratified_selection(
    rows: list[dict[str, Any]],
    selected_count: int,
    seed: int,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select an exact-size, deterministic sample with proportional strata."""
    if selected_count < 0 or selected_count > len(rows):
        raise ValueError(
            f"{label} count must be between 0 and {len(rows)}, got {selected_count}"
        )

    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["primary_domain"], row["difficulty"], row["context_tier"])
        strata[key].append(row)

    allocations = {key: 0 for key in strata}
    targets = {
        key: selected_count * len(values) / len(rows)
        for key, values in strata.items()
    }
    for key, target in targets.items():
        allocations[key] = min(math.floor(target), len(strata[key]))

    remaining = selected_count - sum(allocations.values())
    while remaining:
        eligible = [key for key in strata if allocations[key] < len(strata[key])]
        if not eligible:
            raise RuntimeError(f"Unable to allocate all {selected_count} {label} rows")
        key = max(
            eligible,
            key=lambda item: (
                targets[item] - allocations[item],
                stable_hash(seed, f"{label}:{item}"),
            ),
        )
        allocations[key] += 1
        remaining -= 1

    selected: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    for key, values in strata.items():
        ordered = sorted(
            values,
            key=lambda row: stable_hash(seed, f"{label}:{row['task_id']}"),
        )
        boundary = allocations[key]
        selected.extend(ordered[:boundary])
        remainder.extend(ordered[boundary:])

    return (
        sorted(selected, key=lambda row: row["task_id"]),
        sorted(remainder, key=lambda row: row["task_id"]),
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    policy = tomllib.loads(args.policy.read_text())
    exclusions = policy.get("exclusions", {})
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dirs = sorted(path for path in source_dir.glob("task_*") if path.is_dir())
    if not task_dirs:
        raise SystemExit(f"No task directories found in {source_dir}")

    all_rows = [build_row(task_dir, exclusions) for task_dir in task_dirs]
    stable_rows = [row for row in all_rows if row["stable_candidate"]]
    context_rows = [
        row
        for row in stable_rows
        if row["learnable_by_o3"] and row["context_tier"] in {"medium", "long"}
    ]
    holdout_count = args.validation_count + args.test_count
    eval_rows, train_rows = fixed_stratified_selection(
        stable_rows,
        holdout_count,
        args.seed,
        "holdout",
    )
    validation_rows, test_rows = fixed_stratified_selection(
        eval_rows,
        args.validation_count,
        args.seed + 1,
        "validation",
    )
    context_train_rows, context_eval_rows = stratified_split(
        context_rows, args.eval_fraction, args.seed
    )
    outputs = {
        "all": all_rows,
        "stable": stable_rows,
        "context_rich": context_rows,
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
        "eval": eval_rows,
        "context_train": context_train_rows,
        "context_eval": context_eval_rows,
    }
    split_dir = args.split_dir.resolve()
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
        write_csv(output_dir / f"{name}.csv", rows)
        write_ids(split_dir / f"{name}.txt", rows)

    summary = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "seed": args.seed,
        "validation_count": args.validation_count,
        "test_count": args.test_count,
        "context_eval_fraction": args.eval_fraction,
        "policy": str(args.policy),
        "counts": {
            "all": len(all_rows),
            "stable": len(stable_rows),
            "context_rich": len(context_rows),
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
            "eval": len(eval_rows),
            "context_train": len(context_train_rows),
            "context_eval": len(context_eval_rows),
        },
        "context_tiers": {
            tier: sum(row["context_tier"] == tier for row in all_rows)
            for tier in ("short", "medium", "long", "unknown")
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
