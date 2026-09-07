"""Reusable adapter for ordinary prompt-response tasks.

A benchmark supplies three callables: a data loader, a prompt formatter, and a
reward function. This keeps benchmark code out of the ES scheduler while still
supporting arbitrary scalar rewards and task-specific metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from task_adapter import BatchEvaluation, GenerateFn, SamplingConfig
from token_entropy_utils import summarize_token_entropy


DataLoader = Callable[[Any], tuple[list[Any], list[Any]]]
PromptFormatter = Callable[[Any, Any], Any]
RewardFunction = Callable[[str, Any], float | tuple[float, dict[str, float]]]


@dataclass
class SingleTurnAdapter:
    """Functional adapter for stateless, single-response benchmarks."""

    load_data_fn: DataLoader
    format_prompt: PromptFormatter
    score_response: RewardFunction
    name: str = "single_turn"

    def load_data(self, tokenizer: Any) -> tuple[list[Any], list[Any]]:
        return self.load_data_fn(tokenizer)

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
        prompts = [self.format_prompt(row, tokenizer) for row in rows]
        outputs = generate(prompts, generation_seed, sampling)
        if len(outputs) != len(rows):
            raise RuntimeError(
                f"Generation returned {len(outputs)} outputs for "
                f"{len(rows)} task rows"
            )

        rewards: list[float] = []
        metric_values: dict[str, list[float]] = {}
        trajectories: list[dict[str, Any]] = []
        finish_reasons: dict[str, int] = {}
        for index, (output, row) in enumerate(zip(outputs, rows)):
            completion = output.outputs[0] if output.outputs else None
            text = str(getattr(completion, "text", ""))
            scored = self.score_response(text, row)
            reward, metrics = scored if isinstance(scored, tuple) else (scored, {})
            rewards.append(float(reward))
            for key, value in metrics.items():
                metric_values.setdefault(key, []).append(float(value))
            reason = str(
                getattr(completion, "finish_reason", None) or "missing_output"
            )
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
            if len(trajectories) < trajectory_sample_size:
                trajectories.append(
                    {
                        "iteration": iteration,
                        "perturbation_seed": perturbation_seed,
                        "sample_index": index,
                        "task_id": _row_id(row, index),
                        "response": text,
                        "reward": float(reward),
                        "metrics": metrics,
                    }
                )

        entropy = summarize_token_entropy(
            outputs, require=sampling.track_token_entropy
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
            token_entropy_sum=float(entropy["token_entropy_sum"]),
            token_entropy_count=int(entropy["token_entropy_count"]),
            response_entropy_sum=float(entropy["response_entropy_sum"]),
            response_entropy_square_sum=float(
                entropy["response_entropy_square_sum"]
            ),
            response_entropy_count=int(entropy["response_entropy_count"]),
        )


def _row_id(row: Any, fallback: int) -> Any:
    if isinstance(row, dict):
        return row.get("id", fallback)
    return getattr(row, "id", fallback)
