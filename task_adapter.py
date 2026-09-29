"""Task boundary for the asynchronous ES trainer.

Task adapters own data loading, rollout execution, reward computation, and
task-specific metrics. The trainer owns policy versions, perturbations,
asynchronous scheduling, ES updates, and checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import statistics
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class SamplingConfig:
    """Generation settings supplied by the trainer to a task adapter."""

    max_tokens: int
    temperature: float
    top_p: float
    top_k: int


@dataclass
class BatchEvaluation:
    """Task-independent result of evaluating one temporary policy."""

    rewards: list[float]
    # Metrics are per-task means/rates so independently evaluated shards can
    # be combined with task-count weighting.
    metrics: dict[str, float] = field(default_factory=dict)
    trajectories: list[dict[str, Any]] = field(default_factory=list)
    finish_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def fitness(self) -> float:
        return statistics.fmean(self.rewards) if self.rewards else 0.0

    @property
    def reward_std(self) -> float:
        return statistics.pstdev(self.rewards) if self.rewards else 0.0

    @property
    def successes(self) -> int:
        return sum(reward > 0.0 for reward in self.rewards)

    @property
    def task_count(self) -> int:
        return len(self.rewards)

    @classmethod
    def combine(cls, batches: list["BatchEvaluation"]) -> "BatchEvaluation":
        """Combine independently evaluated shards without changing fitness."""
        if not batches:
            return cls(rewards=[])
        total_tasks = sum(batch.task_count for batch in batches)
        metric_names = set().union(*(batch.metrics for batch in batches))
        metrics = {
            name: (
                sum(
                    batch.metrics.get(name, 0.0) * batch.task_count
                    for batch in batches
                )
                / total_tasks
                if total_tasks
                else 0.0
            )
            for name in metric_names
        }
        finish_reasons: dict[str, int] = {}
        for batch in batches:
            for reason, count in batch.finish_reasons.items():
                finish_reasons[reason] = finish_reasons.get(reason, 0) + count
        return cls(
            rewards=[reward for batch in batches for reward in batch.rewards],
            metrics=metrics,
            trajectories=[
                item for batch in batches for item in batch.trajectories
            ],
            finish_reasons=finish_reasons,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_reward": self.fitness,
            "std_task_reward": self.reward_std,
            "successes": self.successes,
            "task_count": self.task_count,
            "finish_reasons": self.finish_reasons,
            "trajectory_samples": self.trajectories,
            **self.metrics,
        }


GenerateFn = Callable[
    [list[Any], int | list[int], SamplingConfig],
    list[Any],
]


class TaskAdapter(Protocol):
    """Interface between the generic ES trainer and one task family."""

    name: str

    @classmethod
    def add_arguments(cls, parser: Any) -> None: ...

    @classmethod
    def from_args(cls, args: Any) -> "TaskAdapter": ...

    @classmethod
    def validate_args(cls, args: Any, parser: Any) -> None: ...

    def load_data(
        self, tokenizer: Any
    ) -> tuple[list[Any], list[Any]]: ...

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
    ) -> BatchEvaluation: ...


def load_task_adapter_class(spec: str) -> type[TaskAdapter]:
    """Load an adapter class from ``package.module:ClassName``."""
    if ":" not in spec:
        raise ValueError(
            "--task_adapter must use package.module:ClassName syntax"
        )
    module_name, attribute = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, attribute)
    for method in ("add_arguments", "from_args", "validate_args"):
        if not callable(getattr(adapter_class, method, None)):
            raise TypeError(f"Task adapter {spec!r} has no {method}()")
    return adapter_class
