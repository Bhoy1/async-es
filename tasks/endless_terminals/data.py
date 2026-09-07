"""Load fixed Endless Terminals splits for multi-turn ES rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


TRAIN_DATA_PATH = "tasks/endless_terminals/data/skyrl/train.parquet"
EVAL_DATA_PATH = "tasks/endless_terminals/data/skyrl/validation.parquet"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return value if isinstance(value, list) else [value]


def _parse_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    prompt_messages = [dict(message) for message in _as_list(row["prompt"])]
    reward_spec = dict(row.get("reward_spec") or {})
    extra_info = dict(row.get("extra_info") or {})
    task_dir = str(
        extra_info.get("task_dir")
        or reward_spec.get("ground_truth")
        or ""
    )
    if not task_dir:
        raise ValueError(f"Endless Terminals row {index} has no task directory")
    task_id = str(extra_info.get("task_id") or Path(task_dir).name or index)
    question = prompt_messages[-1].get("content", "") if prompt_messages else ""
    if not prompt_messages:
        raise ValueError(f"Endless Terminals row {task_id!r} has no prompt messages")
    return {
        "id": task_id,
        "question": question,
        "target": task_id,
        "solution": task_dir,
        "task_dir": task_dir,
        "prompt_messages": prompt_messages,
        "metadata": extra_info,
    }


def _load_parquet(path: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Loading Endless Terminals parquet requires pandas and pyarrow."
        ) from exc

    rows = pd.read_parquet(path).to_dict(orient="records")
    return [_parse_row(row, index) for index, row in enumerate(rows)]


def get_data(
    *,
    train_data_path: str | None = None,
    eval_data_path: str | None = None,
):
    train_path = train_data_path or TRAIN_DATA_PATH
    eval_path = eval_data_path or EVAL_DATA_PATH
    train_data = _load_parquet(train_path)
    eval_data = _load_parquet(eval_path)
    print(
        f"Loaded Endless Terminals: train={len(train_data)} from {train_path}, "
        f"eval={len(eval_data)} from {eval_path}"
    )
    return train_data, eval_data
