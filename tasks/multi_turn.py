"""Reusable adapter base for stateful, multi-turn tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
import time
from typing import Any, Protocol

from task_adapter import BatchEvaluation, GenerateFn, SamplingConfig


@dataclass(frozen=True)
class TurnResult:
    """Normalized result of applying one model response to a session."""

    observation: str
    reward: float = 0.0
    done: bool = False
    timed_out: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryState:
    """Task-neutral state accumulated across one model trajectory."""

    row: Any
    messages: list[dict[str, Any]]
    token_ids: list[int] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    timed_out: bool = False
    turns: int = 0
    generation_seconds: float = 0.0
    max_token_hits: int = 0
    final_info: dict[str, Any] = field(default_factory=dict)


class SessionBackend(Protocol):
    """Execution backend used by a multi-turn task adapter."""

    def open_sessions(self, rows: list[Any]) -> list[Any]: ...

    def step_sessions(
        self, requests: list[tuple[Any, str]]
    ) -> list[TurnResult]: ...

    def close_sessions(
        self, sessions: list[Any], *, suppress_errors: bool = False
    ) -> None: ...


class MultiTurnAdapter(ABC):
    """Base adapter with batched generation and continuous session refill."""

    name = "multi_turn"

    def __init__(self, args: Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        """Add task-specific arguments in a concrete subclass."""

    @classmethod
    def from_args(cls, args: Namespace) -> "MultiTurnAdapter":
        return cls(args)

    @classmethod
    def validate_args(
        cls, args: Namespace, parser: ArgumentParser
    ) -> None:
        """Validate task-specific arguments in a concrete subclass."""

    @property
    @abstractmethod
    def max_turns(self) -> int: ...

    @property
    @abstractmethod
    def max_input_tokens(self) -> int: ...

    @property
    @abstractmethod
    def session_pool_size(self) -> int: ...

    @property
    def scheduler_debug(self) -> bool:
        return False

    @abstractmethod
    def load_data(self, tokenizer: Any) -> tuple[list[Any], list[Any]]: ...

    @abstractmethod
    def create_session_backend(self) -> SessionBackend: ...

    @abstractmethod
    def initial_messages(self, row: Any) -> list[dict[str, Any]]: ...

    @abstractmethod
    def terminal_response(self) -> str: ...

    def task_id(self, row: Any, fallback: int) -> Any:
        if isinstance(row, dict):
            return row.get("id", fallback)
        return getattr(row, "id", fallback)

    def render_prompt(
        self,
        state: TrajectoryState,
        tokenizer: Any,
    ) -> dict[str, list[int]]:
        rendered = tokenizer.apply_chat_template(
            state.messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        token_ids = tokenizer(rendered)["input_ids"]
        if self.max_input_tokens > 0 and len(token_ids) > self.max_input_tokens:
            token_ids = token_ids[-self.max_input_tokens :]
        return {"prompt_token_ids": token_ids}

    def task_metrics(
        self,
        session: Any,
        state: TrajectoryState,
    ) -> dict[str, float]:
        return {}

    def trajectory_record(
        self,
        session: Any,
        state: TrajectoryState,
    ) -> dict[str, Any]:
        return {
            "messages": state.messages,
            "reward": state.reward,
            "timed_out": state.timed_out,
            "final_info": state.final_info,
        }

    def completion_finish_reason(self, state: TrajectoryState) -> str:
        if state.timed_out:
            return "task_timeout"
        return "task_success" if state.reward > 0.0 else "task_failure"

    def evaluate_batch(
        self,
        rows: list[Any],
        *,
        tokenizer: Any,
        generate: GenerateFn,
        generation_seed: int,
        sampling: SamplingConfig,
        iteration: int,
        perturbation_seed: int,
        trajectory_sample_size: int,
    ) -> BatchEvaluation:
        if not rows:
            return BatchEvaluation(rewards=[])
        if self.max_turns < 1 or self.session_pool_size < 1:
            raise ValueError("max_turns and session_pool_size must be positive")

        backend = self.create_session_backend()
        completed = self._run_pool(
            rows,
            tokenizer=tokenizer,
            generate=generate,
            generation_seed=generation_seed,
            sampling=sampling,
            backend=backend,
            pool_size=self.session_pool_size,
        )
        return self._build_batch_evaluation(
            completed,
            iteration=iteration,
            perturbation_seed=perturbation_seed,
            trajectory_sample_size=trajectory_sample_size,
        )

    def _run_pool(
        self,
        rows: list[Any],
        *,
        tokenizer: Any,
        generate: GenerateFn,
        generation_seed: int,
        sampling: SamplingConfig,
        backend: SessionBackend,
        pool_size: int,
    ) -> list[tuple[Any, TrajectoryState]]:
        completed: list[tuple[Any, TrajectoryState] | None] = [None] * len(rows)
        active: dict[int, tuple[Any, TrajectoryState]] = {}
        next_index = 0
        round_index = 0

        def admit() -> int:
            nonlocal next_index
            count = min(pool_size - len(active), len(rows) - next_index)
            if count <= 0:
                return 0
            indices = list(range(next_index, next_index + count))
            sessions = backend.open_sessions([rows[index] for index in indices])
            if len(sessions) != len(indices):
                raise RuntimeError(
                    f"Backend opened {len(sessions)} sessions for "
                    f"{len(indices)} rows"
                )
            for index, session in zip(indices, sessions):
                active[index] = (
                    session,
                    TrajectoryState(
                        row=rows[index],
                        messages=[
                            dict(message)
                            for message in self.initial_messages(rows[index])
                        ],
                    ),
                )
            next_index += count
            return count

        admit()
        try:
            while active:
                round_index += 1
                active_indices = sorted(active)
                prompts = [
                    self.render_prompt(active[index][1], tokenizer)
                    for index in active_indices
                ]
                request_seeds = [
                    generation_seed
                    + (index // max(1, pool_size))
                    * max(1, pool_size)
                    * self.max_turns
                    + active[index][1].turns
                    for index in active_indices
                ]
                generation_started = time.time()
                generated = generate(prompts, request_seeds, sampling)
                generation_seconds = time.time() - generation_started
                if len(generated) != len(active_indices):
                    raise RuntimeError(
                        f"Generation returned {len(generated)} outputs for "
                        f"{len(active_indices)} active sessions"
                    )

                requests = []
                for index, output in zip(active_indices, generated):
                    session, state = active[index]
                    text, token_ids, finish_reason = _completion_data(output)
                    state.messages.append({"role": "assistant", "content": text})
                    state.token_ids.extend(token_ids)
                    state.turns += 1
                    state.generation_seconds += generation_seconds
                    if (
                        finish_reason == "length"
                        or len(token_ids) >= sampling.max_tokens
                    ):
                        state.max_token_hits += 1
                    requests.append((session, text))

                results = backend.step_sessions(requests)
                if len(results) != len(requests):
                    raise RuntimeError(
                        f"Backend returned {len(results)} results for "
                        f"{len(requests)} session steps"
                    )

                finished = []
                for index, result in zip(active_indices, results):
                    session, state = active[index]
                    _apply_turn_result(state, result)

                forced_indices = [
                    index
                    for index in active_indices
                    if (
                        not active[index][1].done
                        and active[index][1].turns >= self.max_turns
                    )
                ]
                if forced_indices:
                    terminal = self.terminal_response()
                    for index in forced_indices:
                        active[index][1].messages.append(
                            {"role": "assistant", "content": terminal}
                        )
                    forced_results = backend.step_sessions(
                        [(active[index][0], terminal) for index in forced_indices]
                    )
                    if len(forced_results) != len(forced_indices):
                        raise RuntimeError(
                            "Backend returned the wrong number of terminal steps"
                        )
                    for index, result in zip(forced_indices, forced_results):
                        _apply_turn_result(active[index][1], result)
                        if not active[index][1].done:
                            raise RuntimeError(
                                "terminal_response did not finish a session"
                            )

                for index in active_indices:
                    session, state = active[index]
                    if state.done:
                        completed[index] = (session, state)
                        finished.append(index)

                for index in finished:
                    session, _ = active[index]
                    backend.close_sessions([session])
                    del active[index]
                admitted = admit()
                if self.scheduler_debug:
                    print(
                        f"[{self.name} seed={generation_seed} "
                        f"round={round_index}] active={len(active_indices)} "
                        f"completed={len(finished)} admitted={admitted} "
                        f"remaining={len(active)} queued={len(rows) - next_index}",
                        flush=True,
                    )

            return [item for item in completed if item is not None]
        finally:
            backend.close_sessions(
                [session for session, _ in active.values()],
                suppress_errors=True,
            )

    def _build_batch_evaluation(
        self,
        completed: list[tuple[Any, TrajectoryState]],
        *,
        iteration: int,
        perturbation_seed: int,
        trajectory_sample_size: int,
    ) -> BatchEvaluation:
        rewards = [state.reward for _, state in completed]
        metric_values: dict[str, list[float]] = {
            "avg_turns": [float(state.turns) for _, state in completed],
            "avg_output_tokens": [
                float(len(state.token_ids)) for _, state in completed
            ],
        }
        finish_reasons: dict[str, int] = {}
        trajectories = []
        for index, (session, state) in enumerate(completed):
            for key, value in self.task_metrics(session, state).items():
                metric_values.setdefault(key, []).append(float(value))
            reason = self.completion_finish_reason(state)
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
            if len(trajectories) < trajectory_sample_size:
                trajectories.append(
                    {
                        "iteration": iteration,
                        "perturbation_seed": perturbation_seed,
                        "sample_index": index,
                        "task_id": self.task_id(state.row, index),
                        "reward": state.reward,
                        **self.trajectory_record(session, state),
                    }
                )

        return BatchEvaluation(
            rewards=rewards,
            metrics={
                key: sum(values) / len(values)
                for key, values in metric_values.items()
                if values
            },
            trajectories=trajectories,
            finish_reasons=finish_reasons,
        )


def _completion_data(output: Any) -> tuple[str, list[int], str]:
    if not getattr(output, "outputs", None):
        return "", [], "missing_output"
    completion = output.outputs[0]
    text = str(getattr(completion, "text", ""))
    token_ids = list(getattr(completion, "token_ids", None) or [])
    finish_reason = str(
        getattr(completion, "finish_reason", None) or "unknown"
    )
    return (
        text,
        token_ids,
        finish_reason,
    )


def _apply_turn_result(
    state: TrajectoryState,
    result: TurnResult,
) -> None:
    state.messages.append({"role": "user", "content": result.observation})
    state.reward = float(result.reward)
    state.done = bool(result.done)
    state.timed_out = bool(result.timed_out)
    state.final_info = dict(result.info)
