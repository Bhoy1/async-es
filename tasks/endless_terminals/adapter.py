"""Endless Terminals adapter for the generic asynchronous ES trainer."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from task_adapter import BatchEvaluation, GenerateFn, SamplingConfig
from tasks.endless_terminals.data import get_data
from tasks.endless_terminals.rollout import run_endless_rollouts
from token_entropy_utils import summarize_token_entropy


def _reward_for_output(output: Any) -> tuple[float, dict[str, Any]]:
    reward = getattr(output, "precomputed_reward", None)
    if reward is None:
        raise RuntimeError("Endless rollout did not return a sandbox reward")
    return float(reward["reward"]), dict(reward.get("reward_info") or {})


def _output_token_count(output: Any) -> int:
    if not getattr(output, "outputs", None):
        return 0
    return len(getattr(output.outputs[0], "token_ids", None) or [])


def _output_finish_reason(output: Any) -> str:
    if not getattr(output, "outputs", None):
        return "missing_output"
    return str(getattr(output.outputs[0], "finish_reason", None) or "unknown")


class EndlessTerminalsAdapter:
    """Stateful multi-turn task adapter backed by the official environment."""

    name = "endless_terminals"

    def __init__(self, args: Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        group = parser.add_argument_group("Endless Terminals")
        group.add_argument("--endless_official_repo", required=True)
        group.add_argument("--endless_train_data_path", required=True)
        group.add_argument("--endless_eval_data_path", required=True)
        group.add_argument("--endless_max_turns", type=int, default=16)
        group.add_argument("--endless_max_time", type=float, default=300.0)
        group.add_argument(
            "--endless_max_input_tokens", type=int, default=16_384
        )
        group.add_argument(
            "--endless_max_output_chars", type=int, default=50_000
        )
        group.add_argument("--endless_env_batch_size", type=int, default=32)
        group.add_argument("--endless_env_workers", type=int, default=32)
        group.add_argument(
            "--endless_rollout_scheduler",
            choices=["fixed", "continuous"],
            default="continuous",
        )
        group.add_argument("--endless_scheduler_debug", action="store_true")
        group.add_argument("--verbose", action="store_true")

    @classmethod
    def from_args(cls, args: Namespace) -> "EndlessTerminalsAdapter":
        return cls(args)

    @classmethod
    def validate_args(
        cls, args: Namespace, parser: ArgumentParser
    ) -> None:
        for name in (
            "endless_max_turns",
            "endless_env_batch_size",
            "endless_env_workers",
        ):
            if getattr(args, name) < 1:
                parser.error(f"--{name} must be positive")

    def load_data(self, tokenizer: Any) -> tuple[list[Any], list[Any]]:
        return get_data(
            tokenizer,
            train_data_path=self.args.endless_train_data_path,
            eval_data_path=self.args.endless_eval_data_path,
            max_input_tokens=self.args.endless_max_input_tokens,
        )

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
        def generate_turn(prompts, turn_seed):
            return generate(prompts, turn_seed, sampling)

        outputs = run_endless_rollouts(
            rows,
            tokenizer=tokenizer,
            generate_fn=generate_turn,
            official_repo=self.args.endless_official_repo,
            seed=generation_seed,
            max_turns=self.args.endless_max_turns,
            max_time=self.args.endless_max_time,
            max_input_tokens=self.args.endless_max_input_tokens,
            max_tokens_per_turn=sampling.max_tokens,
            max_output_length=self.args.endless_max_output_chars,
            env_batch_size=self.args.endless_env_batch_size,
            env_workers=self.args.endless_env_workers,
            scheduler=self.args.endless_rollout_scheduler,
            scheduler_debug=self.args.endless_scheduler_debug,
            verbose=self.args.verbose,
        )

        rewards: list[float] = []
        turns: list[float] = []
        commands: list[float] = []
        invalid: list[float] = []
        timed_out: list[float] = []
        generated_tokens: list[float] = []
        finish_reasons: dict[str, int] = {}
        trajectories: list[dict[str, Any]] = []
        for index, (output, row) in enumerate(zip(outputs, rows)):
            reward, info = _reward_for_output(output)
            rewards.append(reward)
            turns.append(float(info.get("turns", 0.0)))
            commands.append(float(info.get("command_actions", 0.0)))
            invalid.append(float(info.get("invalid_actions", 0.0)))
            timed_out.append(float(info.get("timed_out", 0.0)))
            generated_tokens.append(float(_output_token_count(output)))
            reason = _output_finish_reason(output)
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
                        "interactive_trajectory": getattr(
                            output, "trajectory", None
                        ),
                    }
                )

        entropy = summarize_token_entropy(
            outputs, require=sampling.track_token_entropy
        )
        return BatchEvaluation(
            rewards=rewards,
            metrics={
                "avg_turns": _mean(turns),
                "avg_commands": _mean(commands),
                "avg_invalid_actions": _mean(invalid),
                "timeout_rate": _mean(timed_out),
                "avg_output_tokens": _mean(generated_tokens),
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
