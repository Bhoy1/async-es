"""Endless Terminals environment sessions for the multi-turn adapter."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import re
import time
from typing import Any

from tasks.multi_turn import TurnResult


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
    """Dependency-light wrapper around the official terminal environment."""

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
        self.sandbox_step_seconds = 0.0
        self.initialized = False
        self.closed = False

    def _initialize(self) -> bool:
        if self.initialized:
            return True
        initialized = self.environment.initialize(run_initial_tests=False)
        self.initialized = bool(initialized)
        return self.initialized

    def step(self, response: str) -> TurnResult:
        if not self._initialize():
            self.close()
            return TurnResult(
                observation="Environment initialization failed.",
                done=True,
                info={"action_type": "init_failed", "grader_output": ""},
            )

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
            step_started = time.time()
            success, output = self.environment.exec(
                str(action["command"] or "")
            )
            self.sandbox_step_seconds += time.time() - step_started
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

        timed_out = time.time() - self.started_at > self.max_time
        if self.turns >= self.max_turns or timed_out:
            done = True

        reward = 0.0
        if done:
            if not timed_out:
                success, grader_output = self.environment.run_final_tests()
                reward = float(bool(success))
            self.close()

        return TurnResult(
            observation=observation,
            reward=reward,
            done=done,
            timed_out=timed_out,
            info={
                "action_type": action_type,
                "grader_output": grader_output,
            },
        )

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.environment.cleanup()
        finally:
            self.initialized = False
            self.closed = True


class LocalEndlessSessionBackend:
    """Execute independent Endless sessions in bounded local threads."""

    def __init__(
        self,
        environment_class,
        *,
        max_turns: int,
        max_time: float,
        max_output_length: int,
        workers: int,
        verbose: bool,
    ) -> None:
        self.environment_class = environment_class
        self.max_turns = max_turns
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.workers = max(1, int(workers))
        self.verbose = verbose

    def open_sessions(
        self,
        task_data: list[dict[str, Any]],
    ) -> list[OfficialEndlessSession]:
        return [
            OfficialEndlessSession(
                item,
                self.environment_class,
                max_turns=self.max_turns,
                max_time=float(
                    item.get("metadata", {}).get("max_time", self.max_time)
                ),
                max_output_length=self.max_output_length,
                verbose=self.verbose,
            )
            for item in task_data
        ]

    def step_sessions(
        self,
        requests: list[tuple[OfficialEndlessSession, str]],
    ) -> list[TurnResult]:
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
