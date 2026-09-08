"""Natural bounded-staleness ES for stateful and single-turn LLM tasks.

Task-specific data, rollout, and reward behavior is supplied by an adapter.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
import wandb

from async_es_coordinator import (
    AlternatingComponentCoordinator,
    BoundedStalenessCoordinator,
)
from distributed_utils import cleanup, launch_engines
from lora_parameterization import (
    LORA_COMPONENTS,
    LoraPolicyState,
    LoraSpec,
    component_for_version,
)
from task_adapter import (
    BatchEvaluation,
    SamplingConfig,
    TaskAdapter,
    load_task_adapter_class,
)


def parse_args() -> argparse.Namespace:
    adapter_parser = argparse.ArgumentParser(add_help=False)
    adapter_parser.add_argument(
        "--task_adapter",
        default=(
            "tasks.endless_terminals.adapter:EndlessTerminalsAdapter"
        ),
    )
    preliminary, _ = adapter_parser.parse_known_args()
    try:
        adapter_class = load_task_adapter_class(preliminary.task_adapter)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        adapter_parser.error(str(exc))

    parser = argparse.ArgumentParser(
        description="Natural bounded-staleness ES for LLM tasks"
    )
    parser.add_argument(
        "--task_adapter",
        required=True,
        help="Task adapter in package.module:ClassName form",
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
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument(
        "--parameterization",
        choices=["full", "lora"],
        default="full",
        help="Optimize all model weights or a persistent alternating LoRA adapter",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument(
        "--lora_target_modules",
        default=(
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
    )
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
    parser.add_argument("--wandb_project", default="async-es")
    parser.add_argument("--wandb_entity")
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )

    adapter_class.add_arguments(parser)
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
    args.lora_target_modules = tuple(
        module.strip()
        for module in args.lora_target_modules.split(",")
        if module.strip()
    )
    if args.sigma is None:
        args.sigma = 0.0075 if args.parameterization == "lora" else 0.0015
    if args.alpha is None:
        args.alpha = 0.005 if args.parameterization == "lora" else 0.00075

    positive = {
        "num_engines": args.num_engines,
        "population_size": args.population_size,
        "num_iterations": args.num_iterations,
        "train_batch_size": args.train_batch_size,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
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
    if args.lora_r < 1 or args.lora_alpha < 1:
        parser.error("--lora_r and --lora_alpha must be positive")
    if args.parameterization == "lora" and not args.lora_target_modules:
        parser.error("--lora_target_modules must not be empty in LoRA mode")
    if args.parameterization == "lora" and args.perturbation_scope != "all":
        parser.error("--perturbation_scope applies only to full parameterization")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu_memory_utilization must be in (0, 1)")
    if bool(args.resume_checkpoint) != bool(args.start_iteration):
        parser.error("resume requires both --resume_checkpoint and --start_iteration")
    if args.start_iteration >= args.num_iterations and not args.eval_only:
        parser.error("--start_iteration must be smaller than --num_iterations")
    if args.eval_only and args.skip_initial_eval:
        parser.error("--eval_only cannot be combined with --skip_initial_eval")
    adapter_class.validate_args(args, parser)
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
    lora_request: LoRARequest | None = None,
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
    return engine.generate.remote(
        prompts,
        sampling,
        use_tqdm=False,
        lora_request=lora_request,
    )


def evaluate_task_batch(
    adapter: TaskAdapter,
    engine,
    rows,
    tokenizer,
    *,
    generation_seed: int,
    sampling: SamplingConfig,
    iteration: int,
    perturbation_seed: int,
    trajectory_sample_size: int,
    lora_request: LoRARequest | None = None,
) -> BatchEvaluation:
    def generate(prompts, seed, config):
        return ray.get(
            generate_rollouts(
                engine,
                prompts,
                seed=seed,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                track_token_entropy=config.track_token_entropy,
                lora_request=lora_request,
            )
        )

    return adapter.evaluate_batch(
        rows,
        tokenizer=tokenizer,
        generate=generate,
        generation_seed=generation_seed,
        sampling=sampling,
        iteration=iteration,
        perturbation_seed=perturbation_seed,
        trajectory_sample_size=trajectory_sample_size,
    )


def sampling_config(
    args: argparse.Namespace, *, evaluation: bool
) -> SamplingConfig:
    prefix = "eval" if evaluation else "train"
    return SamplingConfig(
        max_tokens=args.max_tokens,
        temperature=getattr(args, f"{prefix}_temperature"),
        top_p=getattr(args, f"{prefix}_top_p"),
        top_k=getattr(args, f"{prefix}_top_k"),
        track_token_entropy=args.track_token_entropy,
    )


def select_train_batch(data: list[Any], batch_size: int, seed: int):
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


def serialize_batch_evaluation(result: BatchEvaluation) -> dict[str, Any]:
    return {
        "rewards": result.rewards,
        "metrics": result.metrics,
        "trajectories": result.trajectories,
        "finish_reasons": result.finish_reasons,
        "token_entropy_sum": result.token_entropy_sum,
        "token_entropy_count": result.token_entropy_count,
        "response_entropy_sum": result.response_entropy_sum,
        "response_entropy_square_sum": result.response_entropy_square_sum,
        "response_entropy_count": result.response_entropy_count,
    }


def deserialize_batch_evaluation(payload: dict[str, Any]) -> BatchEvaluation:
    return BatchEvaluation(**payload)


def serialize_pending_record(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "job_id",
        "seed",
        "dispatch_index",
        "dispatch_version",
        "generation_seed",
        "component",
        "engine_index",
        "catchup_seconds",
        "perturb_seconds",
        "rollout_seconds",
        "total_seconds",
        "finished_at",
        "metrics",
    )
    payload = {field: record[field] for field in fields if field in record}
    payload["evaluation"] = serialize_batch_evaluation(record["evaluation"])
    return payload


def deserialize_pending_record(payload: dict[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["evaluation"] = deserialize_batch_evaluation(record["evaluation"])
    return record


def write_checkpoint(
    engines,
    run_dir: Path,
    version: int,
    *,
    lora_state: LoraPolicyState | None = None,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    if lora_state is not None:
        path = (
            run_dir
            / "checkpoints"
            / f"iteration_{version:06d}_lora_adapter"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        lora_state.save(path, trainer_state=trainer_state)
        print(f"Checkpoint saved to {path}", flush=True)
        return path
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
    paths = sorted((run_dir / "checkpoints").glob("iteration_*"))
    parsed = [(int(path.name.split("_")[1]), path) for path in paths]
    rolling = [version for version, _ in parsed if version not in milestones]
    keep = set(milestones) | set(rolling[-max(0, keep_last) :])
    for version, path in parsed:
        if version not in keep:
            if path.is_dir():
                shutil.rmtree(path)
            else:
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
    adapter: TaskAdapter,
    *,
    version: int,
    run,
    run_dir: Path,
    args: argparse.Namespace,
    lora_request: LoRARequest | None = None,
) -> dict[str, Any]:
    data = (
        eval_data[: args.eval_max_samples]
        if args.eval_max_samples > 0
        else eval_data
    )
    started = time.time()
    batches: list[BatchEvaluation] = []
    trajectory_samples_remaining = (
        len(data) if args.eval_trajectory_path else 0
    )
    for offset in range(0, len(data), args.eval_batch_size):
        batch = data[offset : offset + args.eval_batch_size]
        result = evaluate_task_batch(
            adapter,
            engine,
            batch,
            tokenizer,
            generation_seed=args.global_seed + 100_000 + offset,
            sampling=sampling_config(args, evaluation=True),
            iteration=version,
            perturbation_seed=0,
            trajectory_sample_size=trajectory_samples_remaining,
            lora_request=lora_request,
        )
        batches.append(result)
        trajectory_samples_remaining -= len(result.trajectories)
    result = BatchEvaluation.combine(batches)
    metrics = result.to_dict()
    elapsed = time.time() - started
    payload = {
        "version": version,
        "reward": metrics["avg_reward"],
        "successes": metrics["successes"],
        "tasks": metrics["task_count"],
        **result.metrics,
        "token_entropy": metrics["avg_token_entropy"],
        "elapsed_seconds": elapsed,
    }
    write_json(
        run_dir / "evaluations" / f"step_{version:06d}.json", payload
    )
    if args.eval_trajectory_path:
        write_jsonl(
            Path(args.eval_trajectory_path.format(step=version)),
            result.trajectories,
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
        f"task_metrics={result.metrics} "
        f"entropy={metrics['avg_token_entropy']:.4f} time={elapsed:.1f}s",
        flush=True,
    )
    return metrics


def probe_center_entropy(
    engines,
    task_data,
    tokenizer,
    adapter: TaskAdapter,
    *,
    version: int,
    args: argparse.Namespace,
    lora_request: LoRARequest | None = None,
) -> dict[str, Any]:
    shards = [
        task_data[index :: len(engines)] for index in range(len(engines))
    ]
    with ThreadPoolExecutor(max_workers=len(engines)) as executor:
        futures = [
            executor.submit(
                evaluate_task_batch,
                adapter,
                engine,
                shard,
                tokenizer,
                generation_seed=args.global_seed + 200_000 + version + index,
                sampling=sampling_config(args, evaluation=False),
                iteration=version,
                perturbation_seed=0,
                trajectory_sample_size=0,
                lora_request=lora_request,
            )
            for index, (engine, shard) in enumerate(zip(engines, shards))
            if shard
        ]
        batches = [future.result() for future in futures]
    result = BatchEvaluation.combine(batches)
    if result.token_entropy_count == 0:
        raise RuntimeError(
            "Token entropy tracking was requested, but the task adapter "
            "returned no entropy values."
        )
    return result.to_dict()


def run_async_es(
    *,
    args: argparse.Namespace,
    engines,
    train_data,
    eval_data,
    tokenizer,
    adapter: TaskAdapter,
    run,
    run_dir: Path,
    scope_stats: dict[str, Any],
    lora_state: LoraPolicyState | None = None,
    resume_trainer_state: dict[str, Any] | None = None,
) -> None:
    lora_mode = lora_state is not None
    if lora_mode and resume_trainer_state:
        coordinator = AlternatingComponentCoordinator.from_state_dict(
            resume_trainer_state["coordinator"],
            deserialize_pending_record,
        )
        if coordinator.current_version != args.start_iteration:
            raise ValueError(
                "LoRA trainer state version does not match --start_iteration"
            )
    elif lora_mode:
        coordinator = AlternatingComponentCoordinator[dict[str, Any]](
            cohort_size=args.population_size,
            max_staleness=args.max_policy_staleness,
            components=LORA_COMPONENTS,
            initial_version=args.start_iteration,
        )
    else:
        coordinator = BoundedStalenessCoordinator[dict[str, Any]](
            cohort_size=args.population_size,
            max_staleness=args.max_policy_staleness,
            initial_version=args.start_iteration,
        )
    updates: list[dict[str, Any]] = []
    engine_versions = [args.start_iteration] * len(engines)
    dispatch_counts: dict[int, int] = {
        int(version): int(count)
        for version, count in (resume_trainer_state or {})
        .get("dispatch_counts", {})
        .items()
    }
    train_batches: dict[int, list[Any]] = {}
    in_flight: dict[Any, int] = {}
    job_counter = int((resume_trainer_state or {}).get("job_counter", 0))
    completed_total = int(
        (resume_trainer_state or {}).get("completed_total", 0)
    )
    discarded_rollout_seconds = float(
        (resume_trainer_state or {}).get("discarded_rollout_seconds", 0.0)
    )
    run_started = time.time()
    previous_update_time = run_started
    last_barrier_version: int | None = None

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

    def trainer_state_payload() -> dict[str, Any] | None:
        if not lora_mode:
            return None
        return {
            "parameterization": "lora",
            "version": coordinator.current_version,
            "coordinator": coordinator.state_dict(serialize_pending_record),
            "dispatch_counts": {
                str(version): count
                for version, count in dispatch_counts.items()
            },
            "job_counter": job_counter,
            "completed_total": completed_total,
            "discarded_rollout_seconds": discarded_rollout_seconds,
        }

    def center_lora_request(
        version: int,
    ) -> tuple[LoRARequest | None, Path | None]:
        if not lora_mode:
            return None, None
        path = (
            run_dir
            / ".lora_runtime"
            / "centers"
            / f"version_{version:06d}_{time.time_ns()}"
        )
        lora_state.save(path)
        request = LoRARequest(
            f"center-v{version}",
            1_500_000_000 + version,
            str(path),
        )
        return request, path

    def run_member(engine_index: int, job: dict[str, Any]):
        started = time.time()
        catchup_started = time.time()
        if not lora_mode:
            apply_updates(engines[engine_index], job["pending_updates"], args)
        catchup_seconds = time.time() - catchup_started
        perturb_seconds = 0.0

        lora_request = None
        if lora_mode:
            lora_request = LoRARequest(
                f"candidate-{job['job_id']}",
                job["job_id"] + 1,
                job["lora_path"],
            )
        else:
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
            evaluation = evaluate_task_batch(
                adapter,
                engines[engine_index],
                job["task_data"],
                tokenizer,
                generation_seed=job["generation_seed"],
                sampling=sampling_config(args, evaluation=False),
                iteration=job["dispatch_version"] + 1,
                perturbation_seed=job["seed"],
                trajectory_sample_size=(
                    args.trajectory_sample_size
                    if args.trajectory_interval > 0
                    else 0
                ),
                lora_request=lora_request,
            )
        except Exception as exc:
            rollout_error = exc
            evaluation = None
        finally:
            rollout_seconds = time.time() - rollout_started
            if lora_mode:
                lora_state.remove_snapshot(job["lora_path"])
            else:
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
            "evaluation": evaluation,
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
        pending = []
        if not lora_mode:
            pending = [
                update
                for update in updates
                if update["version"] > engine_versions[engine_index]
            ]
        job_counter += 1
        component = component_for_version(version) if lora_mode else "full"
        job = {
            "job_id": job_counter,
            "seed": seed,
            "dispatch_index": dispatch_index,
            "dispatch_version": version,
            "generation_seed": args.global_seed + version + 1,
            "task_data": batch_for(version),
            "pending_updates": pending,
            "component": component,
        }
        if lora_mode:
            candidate_path = (
                run_dir
                / ".lora_runtime"
                / "candidates"
                / f"job_{job_counter:09d}"
            )
            lora_state.save_candidate(
                candidate_path,
                seed=seed,
                sigma=args.sigma,
                component=component,
            )
            job["lora_path"] = str(candidate_path)
        print(
            f"[Job {job_counter}] dispatch engine={engine_index} "
            f"version={version} component={component} seed={seed} "
            f"catchup={len(pending)}",
            flush=True,
        )
        in_flight[executor.submit(run_member, engine_index, job)] = engine_index

    def synchronize_engines() -> float:
        started = time.time()
        if lora_mode:
            engine_versions[:] = [coordinator.current_version] * len(engines)
            return time.time() - started
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
        nonlocal last_barrier_version
        started = time.time()
        sync_seconds = synchronize_engines()
        center_request, center_path = center_lora_request(version)
        checkpoint_due = force_checkpoint or (
            args.checkpoint_interval > 0
            and version % args.checkpoint_interval == 0
        )
        try:
            if checkpoint_due:
                write_checkpoint(
                    engines,
                    run_dir,
                    version,
                    lora_state=lora_state,
                    trainer_state=trainer_state_payload(),
                )
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
                    adapter,
                    version=version,
                    args=args,
                    lora_request=center_request,
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
                    adapter,
                    version=version,
                    run=run,
                    run_dir=run_dir,
                    args=args,
                    lora_request=center_request,
                )
        finally:
            if lora_mode:
                lora_state.remove_snapshot(center_path)
        run.log(
            {
                "async/diagnostic_barrier_seconds": time.time() - started,
                "async/diagnostic_sync_seconds": sync_seconds,
                "async/prefilled_next_cohort": coordinator.cohort_fill,
            },
            step=version,
        )
        previous_update_time = time.time()
        last_barrier_version = version

    def process(
        record: dict[str, Any], *, during_barrier: bool = False
    ) -> bool:
        nonlocal completed_total
        nonlocal discarded_rollout_seconds
        nonlocal previous_update_time
        completed_total += 1
        metrics = record["evaluation"].to_dict()
        record["metrics"] = metrics
        if lora_mode:
            decision = coordinator.observe(
                record,
                component=record["component"],
                dispatch_version=record["dispatch_version"],
            )
        else:
            decision = coordinator.observe(
                record, dispatch_version=record["dispatch_version"]
            )
        if not decision.accepted:
            discarded_rollout_seconds += record["rollout_seconds"]
        print(
            f"[Job {record['job_id']}] "
            f"{'deferred' if getattr(decision, 'deferred', False) else ('accepted' if decision.accepted else 'discarded')}"
            f"{' during barrier' if during_barrier else ''}: "
            f"engine={record['engine_index']} "
            f"dispatch={decision.dispatch_version} "
            f"completion={decision.completion_version} "
            f"staleness={decision.staleness} "
            f"component={record['component']} "
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
        component = cohort[0].component if lora_mode else "full"
        if lora_mode and any(item.component != component for item in cohort):
            raise RuntimeError("LoRA update cohort mixed A and B perturbations")
        updates.append(
            {
                "version": version,
                "seeds": seeds,
                "coeffs": coeffs,
                "component": component,
            }
        )
        if lora_mode:
            lora_state.apply_update(
                seeds=seeds,
                coefficients=coeffs,
                alpha=args.alpha,
                population_size=args.population_size,
                component=component,
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
                    "parameterization": args.parameterization,
                    "lora_component": component if lora_mode else None,
                    "pending_component_results": (
                        coordinator.pending_counts if lora_mode else {}
                    ),
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
                for sample in item["evaluation"].trajectories
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
        if lora_mode:
            train_payload["lora/component_is_a"] = int(component == "A")
            train_payload["lora/component_is_b"] = int(component == "B")
            for name, count in coordinator.pending_counts.items():
                train_payload[f"lora/pending_{name.lower()}"] = count
        task_metric_names = set().union(
            *(item["evaluation"].metrics for item in records)
        )
        for metric_name in task_metric_names:
            train_payload[f"train/task/{metric_name}"] = float(
                np.mean(
                    [
                        item["evaluation"].metrics.get(metric_name, 0.0)
                        for item in records
                    ]
                )
            )
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
        f"engines={len(engines)}, tasks/member={args.train_batch_size}, "
        f"parameterization={args.parameterization}",
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

    if last_barrier_version != coordinator.current_version:
        diagnostic_barrier(coordinator.current_version, force_checkpoint=True)
    if lora_mode:
        final_path = run_dir / "final_lora_adapter"
        lora_state.save(
            final_path,
            trainer_state=trainer_state_payload(),
        )
    else:
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
    adapter_class = load_task_adapter_class(args.task_adapter)
    adapter = adapter_class.from_args(args)
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
        f"{args.model_name.split('/')[-1]}_{adapter.name}_async_"
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
        train_data, eval_data = adapter.load_data(tokenizer)
        print(
            f"Loaded train={len(train_data)}, eval={len(eval_data)}",
            flush=True,
        )
        lora_state = None
        resume_trainer_state = None
        if args.parameterization == "lora":
            lora_spec = LoraSpec(
                rank=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
            )
            if args.resume_checkpoint:
                lora_state = LoraPolicyState.load(
                    args.resume_checkpoint,
                    expected_spec=lora_spec,
                )
                trainer_state_path = (
                    Path(args.resume_checkpoint) / "trainer_state.json"
                )
                if not args.eval_only and not trainer_state_path.is_file():
                    raise ValueError(
                        "resuming LoRA training requires trainer_state.json"
                    )
                if trainer_state_path.is_file():
                    resume_trainer_state = json.loads(
                        trainer_state_path.read_text()
                    )
                if lora_state.model_name != args.model_name:
                    raise ValueError(
                        "LoRA checkpoint base model does not match --model_name: "
                        f"{lora_state.model_name!r} != {args.model_name!r}"
                    )
            else:
                lora_state = LoraPolicyState.initialize(
                    args.model_name,
                    lora_spec,
                    initialization_seed=args.global_seed,
                )
        engines, pgs = launch_engines(
            args.num_engines,
            args.model_name,
            precision=args.precision,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=args.parameterization == "lora",
            max_lora_rank=args.lora_r,
        )
        if lora_state is not None:
            scope_stats = lora_state.stats()
        else:
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
        if args.resume_checkpoint and lora_state is None:
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
            initial_request = None
            initial_path = None
            if lora_state is not None:
                initial_path = (
                    run_dir
                    / ".lora_runtime"
                    / "centers"
                    / f"initial_{args.start_iteration:06d}"
                )
                lora_state.save(initial_path)
                initial_request = LoRARequest(
                    f"initial-v{args.start_iteration}",
                    1_900_000_000 + args.start_iteration,
                    str(initial_path),
                )
            try:
                evaluate(
                    engines[0],
                    eval_data,
                    tokenizer,
                    adapter,
                    version=args.start_iteration,
                    run=run,
                    run_dir=run_dir,
                    args=args,
                    lora_request=initial_request,
                )
            finally:
                if lora_state is not None:
                    lora_state.remove_snapshot(initial_path)
        if not args.eval_only:
            run_async_es(
                args=args,
                engines=engines,
                train_data=train_data,
                eval_data=eval_data,
                tokenizer=tokenizer,
                adapter=adapter,
                run=run,
                run_dir=run_dir,
                scope_stats=scope_stats,
                lora_state=lora_state,
                resume_trainer_state=resume_trainer_state,
            )
    finally:
        cleanup(engines, pgs, run)


if __name__ == "__main__":
    main()
