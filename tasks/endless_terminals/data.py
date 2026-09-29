"""Load fixed Endless Terminals splits for multi-turn ES rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
    if not prompt_messages:
        raise ValueError(f"Endless Terminals row {task_id!r} has no prompt messages")
    return {
        "id": task_id,
        "task_dir": task_dir,
        "prompt_messages": prompt_messages,
        "metadata": extra_info,
    }


def _load_parquet(path: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Loading Endless Terminals parquet requires pyarrow."
        ) from exc

    rows = pq.read_table(path).to_pylist()
    return [_parse_row(row, index) for index, row in enumerate(rows)]


def get_data(
    *,
    train_data_path: str,
    eval_data_path: str,
):
    train_data = _load_parquet(train_data_path)
    eval_data = _load_parquet(eval_data_path)
    print(
        f"Loaded Endless Terminals: train={len(train_data)} from "
        f"{train_data_path}, eval={len(eval_data)} from {eval_data_path}"
    )
    return train_data, eval_data
