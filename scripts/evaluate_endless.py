#!/usr/bin/env python3
"""Evaluate an ES .pt checkpoint or a Hugging Face model on the test split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_REPO = ROOT / "external" / "endless-terminals"
DEFAULT_TEST_DATA = (
    ROOT / "tasks" / "endless_terminals" / "data" / "skyrl" / "test.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the three-seed, 300-task Endless Terminals evaluation. "
            "Use --checkpoint for a native ES .pt file; omit it for a "
            "Hugging Face model or local Hugging Face directory."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=120)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--max-tokens", type=int, default=2_048)
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--env-batch-size", type=int, default=32)
    parser.add_argument("--env-workers", type=int, default=32)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "evaluation"
    )
    parser.add_argument("--label")
    parser.add_argument("--save-trajectories", action="store_true")
    return parser.parse_args()


def load_result(run_dir: Path, step: int) -> dict[str, Any]:
    result_path = run_dir / "evaluations" / f"step_{step:06d}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"evaluation did not write {result_path}")
    return json.loads(result_path.read_text())


def main() -> None:
    args = parse_args()
    if args.checkpoint_step < 0:
        raise SystemExit("--checkpoint-step must be nonnegative")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("--seeds must contain distinct values")

    official_repo = args.official_repo.resolve()
    test_data = args.test_data.resolve()
    if not official_repo.exists():
        raise SystemExit(f"official Endless Terminals checkout not found: {official_repo}")
    if not test_data.exists():
        raise SystemExit(f"test parquet not found: {test_data}")

    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint and not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    label = args.label or (checkpoint.stem if checkpoint else args.model.split("/")[-1])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evaluation_dir = (args.output_dir / f"{label}_{timestamp}").resolve()
    evaluation_dir.mkdir(parents=True, exist_ok=False)

    runs: list[dict[str, Any]] = []
    result_step = args.checkpoint_step if checkpoint else 0
    for seed in args.seeds:
        run_name = f"seed_{seed}"
        run_dir = evaluation_dir / run_name
        command = [
            sys.executable,
            str(ROOT / "train.py"),
            "--task_adapter",
            "tasks.endless_terminals.adapter:EndlessTerminalsAdapter",
            "--model_name",
            args.model,
            "--num_engines",
            "1",
            "--cuda_devices",
            args.cuda_devices,
            "--precision",
            "bfloat16",
            "--max_model_len",
            str(args.max_model_len),
            "--gpu_memory_utilization",
            str(args.gpu_memory_utilization),
            "--max_tokens",
            str(args.max_tokens),
            "--eval_temperature",
            "0.6",
            "--eval_top_p",
            "1.0",
            "--eval_top_k",
            "-1",
            "--eval_max_samples",
            "300",
            "--eval_batch_size",
            "100",
            "--global_seed",
            str(seed),
            "--endless_official_repo",
            str(official_repo),
            # Training data are never consumed in eval-only mode. Reusing the
            # test parquet avoids requiring an unrelated file for evaluation.
            "--endless_train_data_path",
            str(test_data),
            "--endless_eval_data_path",
            str(test_data),
            "--endless_max_turns",
            "16",
            "--endless_max_time",
            "300",
            "--endless_max_input_tokens",
            "16384",
            "--endless_env_batch_size",
            str(args.env_batch_size),
            "--endless_env_workers",
            str(args.env_workers),
            "--experiment_dir",
            str(evaluation_dir),
            "--run_name",
            run_name,
            "--wandb_mode",
            "disabled",
            "--eval_only",
        ]
        if checkpoint:
            command.extend(
                [
                    "--resume_checkpoint",
                    str(checkpoint),
                    "--start_iteration",
                    str(args.checkpoint_step),
                ]
            )
        if args.save_trajectories:
            command.extend(
                [
                    "--eval_trajectory_path",
                    str(run_dir / "trajectories_step_{step:06d}.jsonl"),
                ]
            )

        print(f"Evaluating seed {seed}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        result = load_result(run_dir, result_step)
        runs.append({"seed": seed, **result})

    rewards = [float(run["reward"]) for run in runs]
    summary = {
        "format": "async_es_endless_evaluation_v1",
        "model": args.model,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_step": result_step,
        "dataset": "obiwan96/endless-terminals",
        "dataset_revision": "26ecf78458e7f756e4d06780d5fbf3dd78e91815",
        "split": "test",
        "tasks_per_seed": 300,
        "seeds": args.seeds,
        "runs": runs,
        "mean_reward": statistics.fmean(rewards),
        "sample_std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
    }
    summary_path = evaluation_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"mean={summary['mean_reward']:.4f} "
        f"sample_std={summary['sample_std_reward']:.4f}; {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
