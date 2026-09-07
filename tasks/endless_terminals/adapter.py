"""Endless Terminals implementation of the reusable multi-turn adapter."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from tasks.endless_terminals.data import get_data
from tasks.endless_terminals.rollout import (
    LocalEndlessSessionBackend,
    load_official_environment,
)
from tasks.multi_turn import MultiTurnAdapter, SessionBackend, TrajectoryState


class EndlessTerminalsAdapter(MultiTurnAdapter):
    """Stateful terminal task backed by the official Endless environment."""

    name = "endless_terminals"

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

    @property
    def max_turns(self) -> int:
        return self.args.endless_max_turns

    @property
    def max_input_tokens(self) -> int:
        return self.args.endless_max_input_tokens

    @property
    def session_pool_size(self) -> int:
        return self.args.endless_env_batch_size

    @property
    def rollout_scheduler(self) -> str:
        return self.args.endless_rollout_scheduler

    @property
    def scheduler_debug(self) -> bool:
        return self.args.endless_scheduler_debug

    def load_data(self, tokenizer: Any) -> tuple[list[Any], list[Any]]:
        return get_data(
            train_data_path=self.args.endless_train_data_path,
            eval_data_path=self.args.endless_eval_data_path,
        )

    def create_session_backend(self) -> SessionBackend:
        environment_class = load_official_environment(
            self.args.endless_official_repo
        )
        return LocalEndlessSessionBackend(
            environment_class,
            max_turns=self.max_turns,
            max_time=self.args.endless_max_time,
            max_output_length=self.args.endless_max_output_chars,
            workers=self.args.endless_env_workers,
            verbose=self.args.verbose,
        )

    def initial_messages(self, row: Any) -> list[dict[str, Any]]:
        return [dict(message) for message in row["prompt_messages"]]

    def terminal_response(self) -> str:
        return "<action>done</action>"

    def task_metrics(
        self,
        session: Any,
        state: TrajectoryState,
    ) -> dict[str, float]:
        return {
            "avg_commands": float(session.commands),
            "avg_invalid_actions": float(session.invalid_actions),
            "timeout_rate": float(state.timed_out),
        }

    def trajectory_record(
        self,
        session: Any,
        state: TrajectoryState,
    ) -> dict[str, Any]:
        reward_info = {
            "task": self.name,
            "scalar_reward": state.reward,
            "success": state.reward,
            "turns": float(state.turns),
            "command_actions": float(session.commands),
            "invalid_actions": float(session.invalid_actions),
            "timed_out": float(state.timed_out),
            "generation_seconds": state.generation_seconds,
            "sandbox_step_seconds": float(session.sandbox_step_seconds),
            "max_token_hits": float(state.max_token_hits),
        }
        return {
            "reward_info": reward_info,
            "interactive_trajectory": {
                "task_id": self.task_id(state.row, 0),
                "messages": state.messages,
                "grader_output": state.final_info.get("grader_output", ""),
                **reward_info,
            },
        }

    def completion_finish_reason(self, state: TrajectoryState) -> str:
        if state.timed_out:
            return "endless_timeout"
        return "endless_success" if state.reward > 0.0 else "endless_failure"
