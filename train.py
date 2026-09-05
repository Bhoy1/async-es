"""Natural bounded-staleness ES for Endless Terminals.

This focused trainer contains the real vLLM/Ray execution path used by the
Endless experiments, without unrelated tasks or controlled-lag experiments.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
import wandb

from async_es_coordinator import BoundedStalenessCoordinator
from distributed_utils import cleanup, launch_engines
from tasks.endless_terminals.data import get_data
from tasks.endless_terminals.rollout import run_endless_rollouts
from token_entropy_utils import summarize_token_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Natural bounded-staleness ES on Endless Terminals"
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_engines", type=int, default=4)
    parser.add_argument("--cuda_devices", default="0,1,2,3")
    parser.add_argument(
        "--precision",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--max_model_len", type=int, default=16_384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    parser.add_argument("--population_size", type=int, default=30)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--max_policy_staleness", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.0015)
    parser.add_argument("--alpha", type=float, default=0.00075)
    parser.add_argument(
        "--perturbation_scope", choices=["all", "matrix"], default="all"
    )
    parser.add_argument("--caching", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=42)

    parser.add_argument("--train_temperature", type=float, default=0.6)
    parser.add_argument("--train_top_p", type=float, default=1.0)
    parser.add_argument("--train_top_k", type=int, default=-1)
    parser.add_argument("--eval_temperature", type=float, default=0.6)
    parser.add_argument("--eval_top_p", type=float, default=1.0)
    parser.add_argument("--eval_top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=2_048)

    parser.add_argument("--endless_official_repo", required=True)
    parser.add_argument("--endless_train_data_path", required=True)
    parser.add_argument("--endless_eval_data_path", required=True)
    parser.add_argument("--endless_max_turns", type=int, default=16)
    parser.add_argument("--endless_max_time", type=float, default=300.0)
    parser.add_argument("--endless_max_input_tokens", type=int, default=16_384)
    parser.add_argument("--endless_max_output_chars", type=int, default=50_000)
    parser.add_argument("--endless_env_batch_size", type=int, default=32)
    parser.add_argument("--endless_env_workers", type=int, default=32)
    parser.add_argument(
        "--endless_rollout_scheduler",
        choices=["fixed", "continuous"],
        default="continuous",
    )
    parser.add_argument("--endless_scheduler_debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_max_samples", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--eval_trajectory_path")
    parser.add_argument("--skip_initial_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--track_token_entropy", action="store_true")
    parser.add_argument("--center_entropy_interval", type=int, default=10)

    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--checkpoint_keep_last", type=int, default=3)
    parser.add_argument("--checkpoint_keep_iterations", default="25,50,75,100")
    parser.add_argument("--resume_checkpoint")
    parser.add_argument("--start_iteration", type=int, default=0)
    parser.add_argument("--trajectory_interval", type=int, default=10)
    parser.add_argument("--trajectory_sample_size", type=int, default=8)

    parser.add_argument("--experiment_dir", default="outputs")
    parser.add_argument("--run_name")
    parser.add_argument("--wandb_project", default="async-es-endless")
    parser.add_argument("--wandb_entity")
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    try:
        args.checkpoint_keep_iterations = sorted({
            int(value.strip())
            for value in args.checkpoint_keep_iterations.split(",")
            if value.strip()
        })
    except ValueError:
        parser.error(
            "--checkpoint_keep_iterations must contain comma-separated integers"
        )

    positive = {
        "num_engines": args.num_engines,
        "population_size": args.population_size,
        "num_iterations": args.num_iterations,
        "train_batch_size": args.train_batch_size,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "endless_max_turns": args.endless_max_turns,
        "endless_env_batch_size": args.endless_env_batch_size,
        "endless_env_workers": args.endless_env_workers,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error(f"--{name} must be positive")
    if args.population_size <= args.num_engines:
        parser.error(
            "--population_size must exceed --num_engines so a diagnostic drain "
            "cannot accidentally complete another update cohort"
        )
    if args.max_policy_staleness < 0:
        parser.error("--max_policy_staleness must be non-negative")
    if args.sigma <= 0.0 or args.alpha <= 0.0:
        parser.error("--sigma and --alpha must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu_memory_utilization must be in (0, 1)")
    if bool(args.resume_checkpoint) != bool(args.start_iteration):
        parser.error("resume requires both --resume_checkpoint and --start_iteration")
    if args.start_iteration >= args.num_iterations and not args.eval_only:
        parser.error("--start_iteration must be smaller than --num_iterations")
    if args.eval_only and args.skip_initial_eval:
        parser.error("--eval_only cannot be combined with --skip_initial_eval")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_rollouts(
    engine,
    prompts,
    *,
    seed: int | list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    track_token_entropy: bool,
):
    def params(request_seed: int) -> SamplingParams:
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=int(request_seed),
            max_tokens=max_tokens,
            logprobs=1 if track_token_entropy else None,
        )

    sampling = (
        [params(item) for item in seed] if isinstance(seed, list) else params(seed)
    )
    return engine.generate.remote(prompts, sampling, use_tqdm=False)


def generate_endless_outputs(
    engine,
    task_data: list[dict[str, Any]],
    tokenizer,
    *,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    args: argparse.Namespace,
):
    def generate_fn(prompts, turn_seed):
        return ray.get(
            generate_rollouts(
                engine,
                prompts,
                seed=turn_seed,
                max_tokens=args.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                track_token_entropy=args.track_token_entropy,
            )
        )

    return run_endless_rollouts(
        task_data,
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        official_repo=args.endless_official_repo,
        seed=seed,
        max_turns=args.endless_max_turns,
        max_time=args.endless_max_time,
        max_input_tokens=args.endless_max_input_tokens,
        max_tokens_per_turn=args.max_tokens,
        max_output_length=args.endless_max_output_chars,
        env_batch_size=args.endless_env_batch_size,
        env_workers=args.endless_env_workers,
        scheduler=args.endless_rollout_scheduler,
        scheduler_debug=args.endless_scheduler_debug,
        verbose=args.verbose,
    )


def reward_for_output(output: Any) -> tuple[float, dict[str, Any]]:
    reward = getattr(output, "precomputed_reward", None)
    if reward is None:
        raise RuntimeError("Endless rollout did not return a sandbox reward")
    return float(reward["reward"]), dict(reward.get("reward_info") or {})


def output_token_count(output: Any) -> int:
    if not getattr(output, "outputs", None):
        return 0
    return len(getattr(output.outputs[0], "token_ids", None) or [])


def output_finish_reason(output: Any) -> str:
    if not getattr(output, "outputs", None):
        return "missing_output"
    return str(getattr(output.outputs[0], "finish_reason", None) or "unknown")


def summarize_outputs(
    outputs,
    task_data,
    *,
    iteration: int,
    perturbation_seed: int,
    trajectory_sample_size: int,
    require_entropy: bool,
) -> dict[str, Any]:
    rewards: list[float] = []
    turns: list[float] = []
    commands: list[float] = []
    invalid: list[float] = []
    timed_out: list[float] = []
    generated_tokens: list[int] = []
    finish_reasons: dict[str, int] = {}
    trajectories = []
    for index, (output, row) in enumerate(zip(outputs, task_data)):
        reward, info = reward_for_output(output)
        rewards.append(reward)
        turns.append(float(info.get("turns", 0.0)))
        commands.append(float(info.get("command_actions", 0.0)))
        invalid.append(float(info.get("invalid_actions", 0.0)))
        timed_out.append(float(info.get("timed_out", 0.0)))
        generated_tokens.append(output_token_count(output))
        reason = output_finish_reason(output)
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        if len(trajectories) < trajectory_sample_size:
            trajectories.append(
                {
                    "iteration": iteration,
                    "perturbation_seed": perturbation_seed,
                    "sample_index": index,
                    "task_id": row.get("id"),
                    "reward": reward,
                    "reward_info": info,
                    "interactive_trajectory": getattr(output, "trajectory", None),
                }
            )
    entropy = summarize_token_entropy(outputs, require=require_entropy)
    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_task_reward": float(np.std(rewards)) if rewards else 0.0,
        "successes": int(sum(reward > 0.0 for reward in rewards)),
        "task_count": len(rewards),
        "avg_turns": float(np.mean(turns)) if turns else 0.0,
        "avg_commands": float(np.mean(commands)) if commands else 0.0,
        "avg_invalid_actions": float(np.mean(invalid)) if invalid else 0.0,
        "timeout_rate": float(np.mean(timed_out)) if timed_out else 0.0,
        "avg_output_tokens": (
            float(np.mean(generated_tokens)) if generated_tokens else 0.0
        ),
        "finish_reasons": finish_reasons,
        "trajectory_samples": trajectories,
        **entropy,
    }


def select_train_batch(data: list[dict[str, Any]], batch_size: int, seed: int):
    if batch_size >= len(data):
        return data
    indices = np.random.default_rng(seed).choice(
        len(data), size=batch_size, replace=False
    )
    return [data[int(index)] for index in sorted(indices)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")


def write_checkpoint(engines, run_dir: Path, version: int) -> Path:
    path = (
        run_dir
        / "checkpoints"
        / f"iteration_{version:06d}_model_weights.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ray.get(
        engines[0].collective_rpc.remote(
            "write_weights_to_disk", args=(str(path),)
        )
    )
    print(f"Checkpoint saved to {path}", flush=True)
    return path


def prune_checkpoints(
    run_dir: Path, keep_last: int, milestones: set[int]
) -> None:
    paths = sorted(
        (run_dir / "checkpoints").glob("iteration_*_model_weights.pt")
    )
    parsed = [(int(path.name.split("_")[1]), path) for path in paths]
    rolling = [version for version, _ in parsed if version not in milestones]
    keep = set(milestones) | set(rolling[-max(0, keep_last) :])
    for version, path in parsed:
        if version not in keep:
            path.unlink()
            print(f"Pruned checkpoint {version}: {path}", flush=True)


def apply_updates(
    engine, updates: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    for update in updates:
        ray.get(
            engine.collective_rpc.remote(
                "update_weights_from_seeds",
                args=(
                    update["seeds"],
                    update["coeffs"],
                    args.alpha,
                    args.population_size,
                    args.caching,
                    "none",
                    0.0,
                    args.perturbation_scope,
                ),
            )
        )


def evaluate(
    engine,
    eval_data,
    tokenizer,
    *,
    version: int,
    run,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    data = (
        eval_data[: args.eval_max_samples]
        if args.eval_max_samples > 0
        else eval_data
    )
    started = time.time()
    outputs = []
    for offset in range(0, len(data), args.eval_batch_size):
        batch = data[offset : offset + args.eval_batch_size]
        outputs.extend(
            generate_endless_outputs(
                engine,
                batch,
                tokenizer,
                seed=args.global_seed + 100_000 + offset,
                temperature=args.eval_temperature,
                top_p=args.eval_top_p,
                top_k=args.eval_top_k,
                args=args,
            )
        )
    metrics = summarize_outputs(
        outputs,
        data,
        iteration=version,
        perturbation_seed=0,
        trajectory_sample_size=len(data) if args.eval_trajectory_path else 0,
        require_entropy=args.track_token_entropy,
    )
    elapsed = time.time() - started
    payload = {
        "version": version,
        "reward": metrics["avg_reward"],
        "successes": metrics["successes"],
        "tasks": metrics["task_count"],
        "turns": metrics["avg_turns"],
        "commands": metrics["avg_commands"],
        "invalid_actions": metrics["avg_invalid_actions"],
        "timeout_rate": metrics["timeout_rate"],
        "token_entropy": metrics["avg_token_entropy"],
        "elapsed_seconds": elapsed,
    }
    write_json(
        run_dir / "evaluations" / f"step_{version:06d}.json", payload
    )
    if args.eval_trajectory_path:
        write_jsonl(
            Path(args.eval_trajectory_path.format(step=version)),
            metrics["trajectory_samples"],
        )
    run.log(
        {
            f"eval/{key}": value
            for key, value in payload.items()
            if key != "version"
        },
        step=version,
    )
    print(
        f"[Eval step {version}] {metrics['successes']}/"
        f"{metrics['task_count']} reward={metrics['avg_reward']:.4f} "
        f"turns={metrics['avg_turns']:.2f} "
        f"timeouts={metrics['timeout_rate']:.3f} "
        f"entropy={metrics['avg_token_entropy']:.4f} time={elapsed:.1f}s",
        flush=True,
    )
    return metrics


def probe_center_entropy(
    engines,
    task_data,
    tokenizer,
    *,
    version: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    shards = [
        task_data[index :: len(engines)] for index in range(len(engines))
    ]
    with ThreadPoolExecutor(max_workers=len(engines)) as executor:
        futures = [
            executor.submit(
                generate_endless_outputs,
                engine,
                shard,
                tokenizer,
                seed=args.global_seed + 200_000 + version + index,
                temperature=args.train_temperature,
                top_p=args.train_top_p,
                top_k=args.train_top_k,
                args=args,
            )
            for index, (engine, shard) in enumerate(zip(engines, shards))
            if shard
        ]
        outputs = [
            output for future in futures for output in future.result()
        ]
    return summarize_token_entropy(outputs, require=True)


def run_async_es(
    *,
    args: argparse.Namespace,
    engines,
    train_data,
    eval_data,
    tokenizer,
    run,
    run_dir: Path,
    scope_stats: dict[str, Any],
) -> None:
    coordinator = BoundedStalenessCoordinator[dict[str, Any]](
        cohort_size=args.population_size,
        max_staleness=args.max_policy_staleness,
        initial_version=args.start_iteration,
    )
    updates: list[dict[str, Any]] = []
    engine_versions = [args.start_iteration] * len(engines)
    dispatch_counts: dict[int, int] = {}
    train_batches: dict[int, list[dict[str, Any]]] = {}
    in_flight: dict[Any, int] = {}
    job_counter = 0
    completed_total = 0
    discarded_rollout_seconds = 0.0
    run_started = time.time()
    previous_update_time = run_started

    def batch_for(version: int):
        if version not in train_batches:
            train_batches[version] = select_train_batch(
                train_data,
                args.train_batch_size,
                args.global_seed + version + 1,
            )
        return train_batches[version]

    def next_perturbation_seed(version: int) -> tuple[int, int]:
        index = dispatch_counts.get(version, 0)
        dispatch_counts[version] = index + 1
        sequence = np.random.SeedSequence(
            [args.global_seed, version, index]
        )
        seed = int(
            sequence.generate_state(1, dtype=np.uint32)[0] % (2**30)
        )
        return seed, index

    def run_member(engine_index: int, job: dict[str, Any]):
        started = time.time()
        catchup_started = time.time()
        apply_updates(engines[engine_index], job["pending_updates"], args)
        catchup_seconds = time.time() - catchup_started
        perturb_seconds = 0.0

        point = time.time()
        ray.get(
            engines[engine_index].collective_rpc.remote(
                "perturb_self_weights",
                args=(
                    job["seed"],
                    args.sigma,
                    args.caching,
                    False,
                    args.perturbation_scope,
                ),
            )
        )
        perturb_seconds += time.time() - point
        rollout_error = None
        rollout_started = time.time()
        try:
            outputs = generate_endless_outputs(
                engines[engine_index],
                job["task_data"],
                tokenizer,
                seed=job["generation_seed"],
                temperature=args.train_temperature,
                top_p=args.train_top_p,
                top_k=args.train_top_k,
                args=args,
            )
        except Exception as exc:
            rollout_error = exc
            outputs = None
        finally:
            rollout_seconds = time.time() - rollout_started
            point = time.time()
            ray.get(
                engines[engine_index].collective_rpc.remote(
                    "restore_self_weights",
                    args=(
                        job["seed"],
                        args.sigma,
                        args.caching,
                        False,
                        args.perturbation_scope,
                    ),
                )
            )
            perturb_seconds += time.time() - point
        if rollout_error is not None:
            raise rollout_error
        return {
            **job,
            "engine_index": engine_index,
            "outputs": outputs,
            "catchup_seconds": catchup_seconds,
            "perturb_seconds": perturb_seconds,
            "rollout_seconds": rollout_seconds,
            "total_seconds": time.time() - started,
            "finished_at": time.monotonic(),
        }

    def submit(executor, engine_index: int) -> None:
        nonlocal job_counter
        version = coordinator.current_version
        seed, dispatch_index = next_perturbation_seed(version)
        pending = [
            update
            for update in updates
            if update["version"] > engine_versions[engine_index]
        ]
        job_counter += 1
        job = {
            "job_id": job_counter,
            "seed": seed,
            "dispatch_index": dispatch_index,
            "dispatch_version": version,
            "generation_seed": args.global_seed + version + 1,
            "task_data": batch_for(version),
            "pending_updates": pending,
        }
        print(
            f"[Job {job_counter}] dispatch engine={engine_index} "
            f"version={version} seed={seed} catchup={len(pending)}",
            flush=True,
        )
        in_flight[executor.submit(run_member, engine_index, job)] = engine_index

    def synchronize_engines() -> float:
        started = time.time()
        with ThreadPoolExecutor(max_workers=len(engines)) as executor:
            futures = []
            for index, version in enumerate(engine_versions):
                pending = [
                    update
                    for update in updates
                    if update["version"] > version
                ]
                futures.append(
                    executor.submit(
                        apply_updates, engines[index], pending, args
                    )
                )
            for future in futures:
                future.result()
        engine_versions[:] = [coordinator.current_version] * len(engines)
        ray.get(
            [
                engine.collective_rpc.remote(
                    "broadcast_all_weights", args=(0,)
                )
                for engine in engines
            ]
        )
        return time.time() - started

    def diagnostics_due(version: int) -> bool:
        return any(
            (
                args.checkpoint_interval > 0
                and version % args.checkpoint_interval == 0,
                args.eval_interval > 0
                and (
                    version % args.eval_interval == 0
                    or version == args.num_iterations
                ),
                args.track_token_entropy
                and args.center_entropy_interval > 0
                and version % args.center_entropy_interval == 0,
            )
        )

    def diagnostic_barrier(
        version: int, *, force_checkpoint: bool = False
    ) -> None:
        nonlocal previous_update_time
        started = time.time()
        sync_seconds = synchronize_engines()
        checkpoint_due = force_checkpoint or (
            args.checkpoint_interval > 0
            and version % args.checkpoint_interval == 0
        )
        if checkpoint_due:
            write_checkpoint(engines, run_dir, version)
            prune_checkpoints(
                run_dir,
                args.checkpoint_keep_last,
                args.checkpoint_keep_iterations,
            )
        if (
            args.track_token_entropy
            and args.center_entropy_interval > 0
            and version % args.center_entropy_interval == 0
        ):
            entropy = probe_center_entropy(
                engines,
                batch_for(max(0, version - 1)),
                tokenizer,
                version=version,
                args=args,
            )
            run.log(
                {
                    "train/center_policy_entropy": entropy[
                        "avg_token_entropy"
                    ],
                    "train/center_response_entropy_mean": entropy[
                        "avg_response_entropy"
                    ],
                    "train/center_entropy_token_count": entropy[
                        "token_entropy_count"
                    ],
                },
                step=version,
            )
        if args.eval_interval > 0 and (
            version % args.eval_interval == 0
            or version == args.num_iterations
        ):
            evaluate(
                engines[0],
                eval_data,
                tokenizer,
                version=version,
                run=run,
                run_dir=run_dir,
                args=args,
            )
        run.log(
            {
                "async/diagnostic_barrier_seconds": time.time() - started,
                "async/diagnostic_sync_seconds": sync_seconds,
                "async/prefilled_next_cohort": coordinator.cohort_fill,
            },
            step=version,
        )
        previous_update_time = time.time()

    def process(
        record: dict[str, Any], *, during_barrier: bool = False
    ) -> bool:
        nonlocal completed_total
        nonlocal discarded_rollout_seconds
        nonlocal previous_update_time
        completed_total += 1
        metrics = summarize_outputs(
            record["outputs"],
            record["task_data"],
            iteration=coordinator.current_version + 1,
            perturbation_seed=record["seed"],
            trajectory_sample_size=(
                args.trajectory_sample_size
                if args.trajectory_interval > 0
                else 0
            ),
            require_entropy=args.track_token_entropy,
        )
        record["metrics"] = metrics
        decision = coordinator.observe(
            record, dispatch_version=record["dispatch_version"]
        )
        if not decision.accepted:
            discarded_rollout_seconds += record["rollout_seconds"]
        print(
            f"[Job {record['job_id']}] "
            f"{'accepted' if decision.accepted else 'discarded'}"
            f"{' during barrier' if during_barrier else ''}: "
            f"engine={record['engine_index']} "
            f"dispatch={decision.dispatch_version} "
            f"completion={decision.completion_version} "
            f"staleness={decision.staleness} "
            f"reward={metrics['avg_reward']:.4f} "
            f"cohort={coordinator.cohort_fill}/{args.population_size} "
            f"rollout={record['rollout_seconds']:.1f}s",
            flush=True,
        )
        if not coordinator.cohort_ready():
            return False

        cohort = coordinator.commit_cohort()
        version = coordinator.current_version
        records = [item.payload for item in cohort]
        rewards = np.asarray(
            [item["metrics"]["avg_reward"] for item in records],
            dtype=np.float64,
        )
        mean_reward = float(rewards.mean())
        std_reward = float(rewards.std())
        coeffs = (
            (rewards - mean_reward) / (std_reward + 1e-8)
        ).tolist()
        seeds = [item["seed"] for item in records]
        stalenesses = [item.staleness for item in cohort]
        updates.append(
            {"version": version, "seeds": seeds, "coeffs": coeffs}
        )

        write_json(
            run_dir
            / "iteration_updates"
            / f"iteration_{version:06d}.json",
            {
                "iteration": version,
                "seeds": seeds,
                "update_coefficients": coeffs,
                "metadata": {
                    "execution_mode": "natural_async",
                    "max_policy_staleness": args.max_policy_staleness,
                    "accepted_staleness": stalenesses,
                    "accepted_dispatch_versions": [
                        item.dispatch_version for item in cohort
                    ],
                    "accepted_job_ids": [
                        item["job_id"] for item in records
                    ],
                    "accepted_total": coordinator.accepted_total,
                    "discarded_total": coordinator.discarded_total,
                    "dispatched_total": job_counter,
                    "perturbation_scope": args.perturbation_scope,
                    "active_parameters": scope_stats["active_parameters"],
                },
            },
        )
        if (
            args.trajectory_interval > 0
            and version % args.trajectory_interval == 0
        ):
            trajectories = [
                sample
                for item in records
                for sample in item["metrics"]["trajectory_samples"]
            ]
            write_jsonl(
                run_dir
                / "trajectories"
                / f"iteration_{version:06d}.jsonl",
                trajectories,
            )

        update_seconds = time.time() - previous_update_time
        previous_update_time = time.time()
        train_payload = {
            "train/avg_reward": mean_reward,
            "train/std_reward": std_reward,
            "train/min_reward": float(rewards.min()),
            "train/max_reward": float(rewards.max()),
            "train/turns": float(
                np.mean(
                    [item["metrics"]["avg_turns"] for item in records]
                )
            ),
            "train/timeout_rate": float(
                np.mean(
                    [item["metrics"]["timeout_rate"] for item in records]
                )
            ),
            "async/current_policy_version": version,
            "async/accepted_staleness_mean": float(np.mean(stalenesses)),
            "async/accepted_staleness_max": int(max(stalenesses)),
            "async/accepted_total": coordinator.accepted_total,
            "async/discarded_total": coordinator.discarded_total,
            "async/dispatched_total": job_counter,
            "async/completed_total": completed_total,
            "async/in_flight_at_commit": len(in_flight),
            "async/update_wall_seconds": update_seconds,
            "async/elapsed_wall_seconds": time.time() - run_started,
            "async/discarded_rollout_seconds": discarded_rollout_seconds,
            "async/acceptance_rate": coordinator.accepted_total
            / max(
                1,
                coordinator.accepted_total + coordinator.discarded_total,
            ),
        }
        if args.track_token_entropy:
            token_count = sum(
                item["metrics"]["token_entropy_count"] for item in records
            )
            entropy_sum = sum(
                item["metrics"]["token_entropy_sum"] for item in records
            )
            train_payload["es/population_token_mean_entropy"] = (
                entropy_sum / token_count if token_count else 0.0
            )
            train_payload["es/population_entropy_token_count"] = token_count
        for age in range(max(stalenesses) + 1):
            train_payload[f"async/accepted_staleness_{age}"] = (
                stalenesses.count(age)
            )
        run.log(train_payload, step=version)
        print(
            f"=== Update {version}: reward={mean_reward:.4f}+/-"
            f"{std_reward:.4f} staleness="
            f"{float(np.mean(stalenesses)):.3f}/{max(stalenesses)} "
            f"accepted={coordinator.accepted_total} "
            f"discarded={coordinator.discarded_total} ===",
            flush=True,
        )
        return True

    print(
        f"Natural async ES: versions {args.start_iteration}->"
        f"{args.num_iterations}, cohort={args.population_size}, "
        f"max_staleness={args.max_policy_staleness}, "
        f"engines={len(engines)}, tasks/member={args.train_batch_size}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(engines)) as executor:
        for engine_index in range(len(engines)):
            submit(executor, engine_index)

        while coordinator.current_version < args.num_iterations:
            completed, _ = wait(
                tuple(in_flight), return_when=FIRST_COMPLETED
            )
            records = []
            for future in completed:
                engine_index = in_flight.pop(future)
                record = future.result()
                engine_versions[engine_index] = record["dispatch_version"]
                records.append(record)
            records.sort(key=lambda item: item["finished_at"])
            idle_engines = []
            barrier_version = None
            for record in records:
                idle_engines.append(record["engine_index"])
                if coordinator.current_version >= args.num_iterations:
                    continue
                committed = process(record)
                if (
                    committed
                    and coordinator.current_version < args.num_iterations
                    and diagnostics_due(coordinator.current_version)
                ):
                    barrier_version = coordinator.current_version

            if barrier_version is not None:
                print(
                    f"[Barrier {barrier_version}] draining "
                    f"{len(in_flight)} jobs",
                    flush=True,
                )
                for future, engine_index in list(in_flight.items()):
                    record = future.result()
                    engine_versions[engine_index] = record["dispatch_version"]
                    process(record, during_barrier=True)
                in_flight.clear()
                if coordinator.cohort_ready():
                    raise RuntimeError(
                        "diagnostic drain unexpectedly completed another cohort"
                    )
                diagnostic_barrier(barrier_version)
                idle_engines = list(range(len(engines)))

            if coordinator.current_version < args.num_iterations:
                for engine_index in idle_engines:
                    submit(executor, engine_index)

        for future, engine_index in list(in_flight.items()):
            record = future.result()
            engine_versions[engine_index] = record["dispatch_version"]
            print(
                f"[Job {record['job_id']}] drained after target; "
                "fitness ignored",
                flush=True,
            )
        in_flight.clear()

    diagnostic_barrier(coordinator.current_version, force_checkpoint=True)
    final_path = run_dir / "final_model_weights.pt"
    ray.get(
        engines[0].collective_rpc.remote(
            "write_weights_to_disk", args=(str(final_path),)
        )
    )
    print(
        f"Training complete: updates="
        f"{coordinator.current_version - args.start_iteration}, "
        f"accepted={coordinator.accepted_total}, "
        f"discarded={coordinator.discarded_total}, "
        f"wall={time.time() - run_started:.1f}s; final={final_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.global_seed)
    if args.track_token_entropy:
        marker = os.environ.get(
            "VLLM_ES_TOKEN_ENTROPY_PATCH_MARKER",
            os.path.join(sys.prefix, "VLLM_ES_TOKEN_ENTROPY_PATCH.txt"),
        )
        if not os.path.isfile(marker):
            raise RuntimeError(
                "entropy tracking requires the patched vLLM marker: "
                f"{marker}"
            )
        os.environ["VLLM_ES_TOKEN_ENTROPY"] = "1"

    name = args.run_name or (
        f"{args.model_name.split('/')[-1]}_endless_async_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.experiment_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "args.json", vars(args))
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=name,
        config=vars(args),
        dir=str(run_dir),
        mode=args.wandb_mode,
    )
    engines = []
    pgs = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_data, eval_data = get_data(
            tokenizer,
            train_data_path=args.endless_train_data_path,
            eval_data_path=args.endless_eval_data_path,
            max_input_tokens=args.endless_max_input_tokens,
        )
        print(
            f"Loaded train={len(train_data)}, eval={len(eval_data)}",
            flush=True,
        )
        engines, pgs = launch_engines(
            args.num_engines,
            args.model_name,
            precision=args.precision,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        stats_result = ray.get(
            engines[0].collective_rpc.remote(
                "get_perturbation_scope_stats",
                args=(args.perturbation_scope,),
            )
        )
        scope_stats = (
            stats_result[0]
            if isinstance(stats_result, list)
            else stats_result
        )
        write_json(run_dir / "perturbation_scope.json", scope_stats)
        if args.resume_checkpoint:
            ray.get(
                engines[0].collective_rpc.remote(
                    "load_weights_from_disk",
                    args=(args.resume_checkpoint,),
                )
            )
            ray.get(
                [
                    engine.collective_rpc.remote(
                        "broadcast_all_weights", args=(0,)
                    )
                    for engine in engines
                ]
            )
            print(
                f"Loaded checkpoint {args.resume_checkpoint}", flush=True
            )
        if not args.skip_initial_eval:
            evaluate(
                engines[0],
                eval_data,
                tokenizer,
                version=args.start_iteration,
                run=run,
                run_dir=run_dir,
                args=args,
            )
        if not args.eval_only:
            run_async_es(
                args=args,
                engines=engines,
                train_data=train_data,
                eval_data=eval_data,
                tokenizer=tokenizer,
                run=run,
                run_dir=run_dir,
                scope_stats=scope_stats,
            )
    finally:
        cleanup(engines, pgs, run)


if __name__ == "__main__":
    main()
