"""Official-compatible multi-turn Endless Terminals rollouts for ES."""

from __future__ import annotations

import importlib.util
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MAX_OUTPUT_LENGTH = 50_000
DONE_RE = re.compile(r"<action>\s*done\s*</action>", flags=re.IGNORECASE)
CMD_RE = re.compile(
    r"<command>\s*(.*?)\s*</command>",
    flags=re.IGNORECASE | re.DOTALL,
)
INVALID_ACTION_OBSERVATION = (
    "Could not parse a single <command>...</command> or "
    "<action>done</action>. Please respond with exactly one of those."
)


@dataclass
class EndlessCompletionOutput:
    text: str
    token_ids: list[int]
    logprobs: list[Any] | None
    finish_reason: str


@dataclass
class EndlessRequestOutput:
    outputs: list[EndlessCompletionOutput]
    precomputed_reward: dict[str, Any]
    trajectory: dict[str, Any]


def extract_action(response: str) -> dict[str, str | None]:
    """Match the pinned official parser, including done precedence."""
    if DONE_RE.search(response):
        return {"type": "done", "command": None}
    matches = CMD_RE.findall(response)
    if matches:
        command = matches[-1].strip()
        if command.lower() == "done":
            return {"type": "done", "command": None}
        return {"type": "command", "command": command}
    return {"type": "invalid", "command": None}


def load_official_environment(official_repo: str | Path):
    module_path = Path(official_repo).resolve() / "generator/env.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"official Endless Terminals environment is missing: {module_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "endless_terminals_official_env",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load official environment: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InteractiveContainerEnvironment


class OfficialEndlessSession:
    """Small dependency-free equivalent of the official SkyRL text environment."""

    def __init__(
        self,
        task_data: dict[str, Any],
        environment_class,
        *,
        max_turns: int,
        max_time: float,
        max_output_length: int,
        verbose: bool,
    ) -> None:
        task_dir = Path(task_data["task_dir"])
        self.environment = environment_class(
            container_sif_path=task_dir / "container.sif",
            initial_test_path=task_dir / "test_initial_state.py",
            final_test_path=task_dir / "test_final_state.py",
            def_path=task_dir / "container.def",
            max_actions=max_turns,
            verbose=verbose,
        )
        self.max_turns = max_turns
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.started_at = time.time()
        self.turns = 0
        self.commands = 0
        self.invalid_actions = 0
        self.initialized = False
        self.closed = False

    def _initialize(self) -> bool:
        if self.initialized:
            return True
        initialized = self.environment.initialize(run_initial_tests=False)
        self.initialized = bool(initialized)
        return self.initialized

    def step(self, response: str) -> dict[str, Any]:
        if not self._initialize():
            self.close()
            return {
                "observation": "Environment initialization failed.",
                "reward": 0.0,
                "done": True,
                "timed_out": False,
                "action_type": "init_failed",
                "grader_output": "",
            }

        self.turns += 1
        action = extract_action(response)
        action_type = str(action["type"])
        done = False
        grader_output = ""

        if action_type == "done":
            observation = "Done"
            done = True
        elif action_type == "command":
            self.commands += 1
            success, output = self.environment.exec(str(action["command"] or ""))
            original_length = len(output)
            truncated = ""
            if original_length > self.max_output_length:
                output = output[: self.max_output_length]
                truncated = (
                    "\n[Output truncated: showing first "
                    f"{self.max_output_length} of {original_length} characters]"
                )
            status = "successfully" if success else "failed"
            observation = (
                f"Command executed {status}. Output: {output}{truncated}\n\n"
                f"(exit_code={0 if success else 1})"
            )
        else:
            self.invalid_actions += 1
            observation = INVALID_ACTION_OBSERVATION

        elapsed = time.time() - self.started_at
        timed_out = elapsed > self.max_time
        if self.turns >= self.max_turns or timed_out:
            done = True

        reward = 0.0
        if done:
            if not timed_out:
                success, grader_output = self.environment.run_final_tests()
                reward = float(bool(success))
            self.close()

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "timed_out": timed_out,
            "action_type": action_type,
            "grader_output": grader_output,
        }

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.environment.cleanup()
        finally:
            self.initialized = False
            self.closed = True


class LocalEndlessSessionBackend:
    """Execute session operations in bounded local threads."""

    def __init__(self, environment_class, *, workers: int) -> None:
        self.environment_class = environment_class
        self.workers = max(1, int(workers))

    def open_sessions(
        self,
        task_data: list[dict[str, Any]],
        *,
        max_turns: int,
        max_time: float,
        max_output_length: int,
        verbose: bool,
    ) -> list[OfficialEndlessSession]:
        return [
            OfficialEndlessSession(
                item,
                self.environment_class,
                max_turns=max_turns,
                max_time=float(item.get("metadata", {}).get("max_time", max_time)),
                max_output_length=max_output_length,
                verbose=verbose,
            )
            for item in task_data
        ]

    def step_sessions(
        self,
        requests: list[tuple[OfficialEndlessSession, str]],
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        with ThreadPoolExecutor(
            max_workers=max(1, min(self.workers, len(requests)))
        ) as executor:
            return list(
                executor.map(
                    lambda request: request[0].step(request[1]),
                    requests,
                )
            )

    def close_sessions(
        self,
        sessions: list[OfficialEndlessSession],
        *,
        suppress_errors: bool = False,
    ) -> None:
        errors = []
        for session in sessions:
            try:
                session.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            print(
                "WARNING: failed to close "
                f"{len(errors)} local Endless sessions; "
                f"first_error={type(errors[0]).__name__}: {errors[0]}",
                flush=True,
            )
            if not suppress_errors:
                raise RuntimeError(
                    f"Failed to close {len(errors)} local Endless sessions"
                ) from errors[0]


def _render_context(
    tokenizer,
    messages: list[dict[str, str]],
    max_input_tokens: int,
) -> dict[str, list[int]]:
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    token_ids = tokenizer(rendered)["input_ids"]
    if max_input_tokens > 0 and len(token_ids) > max_input_tokens:
        token_ids = token_ids[-max_input_tokens:]
    return {"prompt_token_ids": token_ids}


def _completion_data(output: Any) -> tuple[str, list[int], list[Any] | None, str]:
    if not getattr(output, "outputs", None):
        return "", [], None, "missing_output"
    completion = output.outputs[0]
    text = str(getattr(completion, "text", ""))
    token_ids = list(getattr(completion, "token_ids", None) or [])
    logprobs = getattr(completion, "logprobs", None)
    finish_reason = str(getattr(completion, "finish_reason", None) or "unknown")
    return text, token_ids, list(logprobs) if logprobs is not None else None, finish_reason


def _new_rollout_state(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [dict(message) for message in item["prompt_messages"]],
        "assistant_responses": [],
        "token_ids": [],
        "logprobs": [],
        "has_logprobs": False,
        "finish_reasons": [],
        "reward": 0.0,
        "timed_out": False,
        "grader_output": "",
        "done": False,
        "generation_seconds": 0.0,
        "max_token_hits": 0,
    }


def _build_request_output(
    item: dict[str, Any],
    session: OfficialEndlessSession,
    state: dict[str, Any],
) -> EndlessRequestOutput:
    reward = float(state["reward"])
    finish_reason = (
        "endless_timeout"
        if state["timed_out"]
        else "endless_success"
        if reward > 0.0
        else "endless_failure"
    )
    reward_info = {
        "task": "endless_terminals",
        "scalar_reward": reward,
        "success": reward,
        "turns": float(session.turns),
        "command_actions": float(session.commands),
        "invalid_actions": float(session.invalid_actions),
        "timed_out": float(state["timed_out"]),
        "generation_seconds": float(state["generation_seconds"]),
        "max_token_hits": float(state["max_token_hits"]),
    }
    if hasattr(session, "sandbox_step_seconds"):
        reward_info["sandbox_step_seconds"] = float(
            session.sandbox_step_seconds
        )
    completion = EndlessCompletionOutput(
        text=(
            state["assistant_responses"][-1]
            if state["assistant_responses"]
            else ""
        ),
        token_ids=list(state["token_ids"]),
        logprobs=(
            list(state["logprobs"])
            if state["has_logprobs"]
            else None
        ),
        finish_reason=finish_reason,
    )
    return EndlessRequestOutput(
        outputs=[completion],
        precomputed_reward={
            "reward": reward,
            "reward_vector": [reward],
            "candidate_reward_vectors": [[reward]],
            "reward_info": reward_info,
        },
        trajectory={
            "task_id": item["id"],
            "messages": state["messages"],
            "grader_output": state["grader_output"],
            **reward_info,
        },
    )


def _run_chunk(
    task_data: list[dict[str, Any]],
    *,
    tokenizer,
    generate_fn: Callable[[list[dict[str, list[int]]], int], list[Any]],
    environment_class,
    seed: int,
    max_turns: int,
    max_time: float,
    max_input_tokens: int,
    max_tokens_per_turn: int,
    max_output_length: int,
    env_workers: int,
    verbose: bool,
    scheduler_debug: bool = False,
    session_backend=None,
) -> list[EndlessRequestOutput]:
    backend = session_backend or LocalEndlessSessionBackend(
        environment_class,
        workers=env_workers,
    )
    sessions = backend.open_sessions(
        task_data,
        max_turns=max_turns,
        max_time=max_time,
        max_output_length=max_output_length,
        verbose=verbose,
    )
    states = [_new_rollout_state(item) for item in task_data]
    rounds = 0
    active_slots = 0

    try:
        for turn in range(max_turns):
            active = [index for index, state in enumerate(states) if not state["done"]]
            if not active:
                break
            rounds += 1
            active_slots += len(active)
            prompts = [
                _render_context(
                    tokenizer,
                    states[index]["messages"],
                    max_input_tokens,
                )
                for index in active
            ]
            generation_started = time.time()
            generated = generate_fn(prompts, seed + turn)
            generation_seconds = time.time() - generation_started
            if len(generated) != len(active):
                raise RuntimeError(
                    f"Endless generation returned {len(generated)} outputs for "
                    f"{len(active)} active sessions"
                )

            responses = []
            for index, output in zip(active, generated):
                text, token_ids, logprobs, finish_reason = _completion_data(output)
                state = states[index]
                state["messages"].append({"role": "assistant", "content": text})
                state["assistant_responses"].append(text)
                state["token_ids"].extend(token_ids)
                state["finish_reasons"].append(finish_reason)
                if (
                    finish_reason == "length"
                    or len(token_ids) >= max_tokens_per_turn
                ):
                    state["max_token_hits"] += 1
                state["generation_seconds"] += generation_seconds
                if logprobs is not None:
                    state["has_logprobs"] = True
                    state["logprobs"].extend(logprobs)
                responses.append((index, text))

            step_started = time.time()
            step_results = backend.step_sessions(
                [(sessions[index], response) for index, response in responses]
            )
            step_seconds = time.time() - step_started

            completed_this_turn = 0
            for (index, _), result in zip(responses, step_results):
                state = states[index]
                state["messages"].append(
                    {"role": "user", "content": str(result["observation"])}
                )
                state["reward"] = float(result["reward"])
                state["timed_out"] = bool(result["timed_out"])
                state["grader_output"] = str(result["grader_output"])
                state["done"] = bool(result["done"])
                completed_this_turn += int(state["done"])

            if scheduler_debug:
                remaining = sum(not state["done"] for state in states)
                print(
                    f"[Endless fixed seed={seed} round={turn + 1}] "
                    f"active={len(active)} completed={completed_this_turn} "
                    f"remaining={remaining} generate={generation_seconds:.3f}s "
                    f"sandbox={step_seconds:.3f}s",
                    flush=True,
                )

        for index, state in enumerate(states):
            if not state["done"]:
                result = backend.step_sessions(
                    [(sessions[index], "<action>done</action>")]
                )[0]
                state["messages"].append(
                    {"role": "assistant", "content": "<action>done</action>"}
                )
                state["messages"].append(
                    {"role": "user", "content": str(result["observation"])}
                )
                state["reward"] = float(result["reward"])
                state["timed_out"] = bool(result["timed_out"])
                state["grader_output"] = str(result["grader_output"])
                state["done"] = True
    finally:
        backend.close_sessions(sessions)

    outputs = [
        _build_request_output(item, session, state)
        for item, session, state in zip(task_data, sessions, states)
    ]
    if scheduler_debug:
        turn_counts: dict[int, int] = {}
        for session in sessions:
            turn_counts[session.turns] = turn_counts.get(session.turns, 0) + 1
        slot_utilization = (
            active_slots / (rounds * len(task_data))
            if rounds > 0 and task_data
            else 0.0
        )
        print(
            f"[Endless fixed seed={seed}] summary "
            f"rounds={rounds} tasks={len(task_data)} "
            f"slot_utilization={slot_utilization:.3f} "
            f"turn_histogram={dict(sorted(turn_counts.items()))}",
            flush=True,
        )
    return outputs


def _run_continuous_pool(
    task_data: list[dict[str, Any]],
    *,
    tokenizer,
    generate_fn: Callable[[list[dict[str, list[int]]], list[int]], list[Any]],
    environment_class,
    seed: int,
    max_turns: int,
    max_time: float,
    max_input_tokens: int,
    max_tokens_per_turn: int,
    max_output_length: int,
    pool_size: int,
    env_workers: int,
    verbose: bool,
    scheduler_debug: bool = False,
    session_backend=None,
) -> list[EndlessRequestOutput]:
    backend = session_backend or LocalEndlessSessionBackend(
        environment_class,
        workers=env_workers,
    )
    outputs: list[EndlessRequestOutput | None] = [None] * len(task_data)
    active: dict[int, tuple[Any, dict[str, Any]]] = {}
    next_index = 0

    def admit_sessions() -> list[int]:
        nonlocal next_index
        admitted = []
        capacity = min(pool_size - len(active), len(task_data) - next_index)
        if capacity <= 0:
            return admitted
        indices = list(range(next_index, next_index + capacity))
        items = [task_data[index] for index in indices]
        sessions = backend.open_sessions(
            items,
            max_turns=max_turns,
            max_time=max_time,
            max_output_length=max_output_length,
            verbose=verbose,
        )
        for index, item, session in zip(indices, items, sessions):
            active[index] = (session, _new_rollout_state(item))
            admitted.append(index)
        next_index += capacity
        return admitted

    initial_admitted = admit_sessions()
    round_index = 0
    active_slots = 0
    total_refills = 0
    if scheduler_debug:
        print(
            f"[Endless continuous seed={seed}] initialized "
            f"active={len(active)} admitted={len(initial_admitted)} "
            f"queued={len(task_data) - next_index}",
            flush=True,
        )
    try:
        while active:
            round_index += 1
            active_indices = sorted(active)
            active_slots += len(active_indices)
            prompts = [
                _render_context(
                    tokenizer,
                    active[index][1]["messages"],
                    max_input_tokens,
                )
                for index in active_indices
            ]
            request_seeds = [
                seed
                + (index // pool_size) * pool_size * max_turns
                + active[index][0].turns
                for index in active_indices
            ]
            generation_started = time.time()
            generated = generate_fn(prompts, request_seeds)
            generation_seconds = time.time() - generation_started
            if len(generated) != len(active_indices):
                raise RuntimeError(
                    f"Endless generation returned {len(generated)} outputs for "
                    f"{len(active_indices)} active sessions"
                )

            responses = []
            for index, output in zip(active_indices, generated):
                session, state = active[index]
                text, token_ids, logprobs, finish_reason = _completion_data(output)
                state["messages"].append({"role": "assistant", "content": text})
                state["assistant_responses"].append(text)
                state["token_ids"].extend(token_ids)
                state["finish_reasons"].append(finish_reason)
                if (
                    finish_reason == "length"
                    or len(token_ids) >= max_tokens_per_turn
                ):
                    state["max_token_hits"] += 1
                state["generation_seconds"] += generation_seconds
                if logprobs is not None:
                    state["has_logprobs"] = True
                    state["logprobs"].extend(logprobs)
                responses.append((index, text, session))

            step_started = time.time()
            step_results = backend.step_sessions(
                [(session, response) for _, response, session in responses]
            )
            step_seconds = time.time() - step_started

            completed_indices = []
            for (index, _, session), result in zip(responses, step_results):
                state = active[index][1]
                state["messages"].append(
                    {"role": "user", "content": str(result["observation"])}
                )
                state["reward"] = float(result["reward"])
                state["timed_out"] = bool(result["timed_out"])
                state["grader_output"] = str(result["grader_output"])
                state["done"] = bool(result["done"])
                if state["done"]:
                    outputs[index] = _build_request_output(
                        task_data[index],
                        session,
                        state,
                    )
                    completed_indices.append(index)

            for index in completed_indices:
                del active[index]
            admitted_indices = admit_sessions()
            total_refills += len(admitted_indices)
            if scheduler_debug:
                active_turns = [active[index][0].turns for index in active]
                turn_summary = (
                    f"{min(active_turns)}/"
                    f"{sum(active_turns) / len(active_turns):.1f}/"
                    f"{max(active_turns)}"
                    if active_turns
                    else "done"
                )
                print(
                    f"[Endless continuous seed={seed} round={round_index}] "
                    f"active_before={len(active_indices)} "
                    f"completed={len(completed_indices)} "
                    f"admitted={len(admitted_indices)} "
                    f"active_after={len(active)} "
                    f"queued={len(task_data) - next_index} "
                    f"turns_after[min/mean/max]={turn_summary} "
                    f"generate={generation_seconds:.3f}s "
                    f"sandbox={step_seconds:.3f}s",
                    flush=True,
                )
    finally:
        # Cleanup must not replace the rollout exception that triggered it.
        backend.close_sessions(
            [session for session, _ in active.values()],
            suppress_errors=True,
        )

    if any(output is None for output in outputs):
        raise RuntimeError("continuous Endless scheduler did not complete all tasks")
    completed_outputs = [output for output in outputs if output is not None]
    if scheduler_debug:
        turn_counts: dict[int, int] = {}
        for output in completed_outputs:
            turns = int(output.precomputed_reward["reward_info"]["turns"])
            turn_counts[turns] = turn_counts.get(turns, 0) + 1
        slot_utilization = (
            active_slots / (round_index * pool_size)
            if round_index > 0
            else 0.0
        )
        print(
            f"[Endless continuous seed={seed}] summary "
            f"rounds={round_index} tasks={len(task_data)} "
            f"refills={total_refills} "
            f"slot_utilization={slot_utilization:.3f} "
            f"turn_histogram={dict(sorted(turn_counts.items()))}",
            flush=True,
        )
    return completed_outputs


def run_endless_rollouts(
    task_data: list[dict[str, Any]],
    *,
    tokenizer,
    generate_fn: Callable[
        [list[dict[str, list[int]]], int | list[int]],
        list[Any],
    ],
    official_repo: str | Path,
    seed: int,
    max_turns: int = 16,
    max_time: float = 300.0,
    max_input_tokens: int = 16_384,
    max_tokens_per_turn: int = 2_048,
    max_output_length: int = MAX_OUTPUT_LENGTH,
    env_batch_size: int = 8,
    env_workers: int = 8,
    scheduler: str = "fixed",
    scheduler_debug: bool = False,
    verbose: bool = False,
) -> list[EndlessRequestOutput]:
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if env_batch_size < 1 or env_workers < 1:
        raise ValueError("env_batch_size and env_workers must be positive")
    if scheduler not in {"fixed", "continuous"}:
        raise ValueError("scheduler must be 'fixed' or 'continuous'")
    if not task_data:
        return []
    environment_class = load_official_environment(official_repo)
    backend = LocalEndlessSessionBackend(
        environment_class,
        workers=env_workers,
    )
    if scheduler == "continuous":
        started = time.time()
        print(
            f"Endless continuous rollout starting: tasks={len(task_data)}, "
            f"active_pool={env_batch_size}, workers={env_workers}",
            flush=True,
        )
        outputs = _run_continuous_pool(
            task_data,
            tokenizer=tokenizer,
            generate_fn=generate_fn,
            environment_class=environment_class,
            seed=seed,
            max_turns=max_turns,
            max_time=max_time,
            max_input_tokens=max_input_tokens,
            max_tokens_per_turn=max_tokens_per_turn,
            max_output_length=max_output_length,
            pool_size=env_batch_size,
            env_workers=env_workers,
            verbose=verbose,
            scheduler_debug=scheduler_debug,
            session_backend=backend,
        )
        print(
            f"Endless continuous rollout finished: "
            f"reward={sum(output.precomputed_reward['reward'] for output in outputs) / len(outputs):.3f}, "
            f"time={time.time() - started:.2f}s",
            flush=True,
        )
        return outputs

    outputs = []
    for start in range(0, len(task_data), env_batch_size):
        chunk = task_data[start : start + env_batch_size]
        chunk_started = time.time()
        print(
            f"Endless rollout chunk {start + 1}-{start + len(chunk)}/"
            f"{len(task_data)} starting",
            flush=True,
        )
        chunk_outputs = _run_chunk(
            chunk,
            tokenizer=tokenizer,
            generate_fn=generate_fn,
            environment_class=environment_class,
            seed=seed + start * max_turns,
            max_turns=max_turns,
            max_time=max_time,
            max_input_tokens=max_input_tokens,
            max_tokens_per_turn=max_tokens_per_turn,
            max_output_length=max_output_length,
            env_workers=env_workers,
            verbose=verbose,
            scheduler_debug=scheduler_debug,
            session_backend=backend,
        )
        outputs.extend(chunk_outputs)
        print(
            f"Endless rollout chunk {start + 1}-{start + len(chunk)}/"
            f"{len(task_data)} finished: "
            f"reward={sum(output.precomputed_reward['reward'] for output in chunk_outputs) / len(chunk_outputs):.3f}, "
            f"time={time.time() - chunk_started:.2f}s",
            flush=True,
        )
    return outputs
