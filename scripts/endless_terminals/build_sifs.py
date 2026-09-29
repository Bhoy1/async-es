#!/usr/bin/env python3
"""Build Endless Terminals Apptainer images from the downloaded task sources."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE_IMAGE_REFERENCE = "docker://ubuntu:22.04"
LOCAL_BASE_REFERENCE = "./ubuntu_22.04.sif"
DOCKER_BOOTSTRAP = "Bootstrap: docker\nFrom: ubuntu:22.04"
LOCAL_BOOTSTRAP_RE = re.compile(
    r"^bootstrap:\s*localimage[ \t]*\n"
    r"from:\s*\./ubuntu_22\.04\.sif[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
TMP_CHMOD_RE = re.compile(
    r"^[ \t]*chmod[ \t]+0?1777[ \t]+/tmp[ \t]*(?:#.*)?$\n?",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("tasks/endless_terminals/data/source"),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("tasks/endless_terminals/splits/train.txt"),
    )
    parser.add_argument(
        "--base-sif",
        type=Path,
        default=Path("tasks/endless_terminals/data/ubuntu_22.04.sif"),
    )
    parser.add_argument(
        "--pull-base",
        action="store_true",
        help="Pull docker://ubuntu:22.04 when --base-sif does not exist.",
    )
    parser.add_argument(
        "--direct-docker-base",
        action="store_true",
        help="Build directly from docker://ubuntu:22.04 instead of a local base SIF.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fakeroot",
        action="store_true",
        help="Build with Singularity/Apptainer fakeroot on unprivileged clusters.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/endless_sif_build"),
    )
    return parser.parse_args()


def ensure_base_image(runtime: str, base_sif: Path, pull_base: bool) -> None:
    if base_sif.exists():
        return
    if not pull_base:
        raise SystemExit(
            f"Base image is missing: {base_sif}. Re-run with --pull-base."
        )
    base_sif.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [runtime, "pull", str(base_sif), BASE_IMAGE_REFERENCE],
        check=True,
    )


def make_portable_definition(
    definition_text: str,
    base_sif: Path,
    direct_docker_base: bool,
) -> str:
    if LOCAL_BOOTSTRAP_RE.search(definition_text) is None:
        raise ValueError("definition does not contain the expected local bootstrap")

    bootstrap = (
        DOCKER_BOOTSTRAP
        if direct_docker_base
        else f"Bootstrap: localimage\nFrom: {base_sif.resolve()}"
    )
    portable_definition = LOCAL_BOOTSTRAP_RE.sub(
        bootstrap,
        definition_text,
        count=1,
    )
    # The sanitized local base already has a sticky, world-writable /tmp.
    # SingularityCE protects that mount point during %post, so chmod fails.
    return TMP_CHMOD_RE.sub("", portable_definition)


def build_one(
    runtime: str,
    source_dir: Path,
    base_sif: Path,
    log_dir: Path,
    task_id: str,
    force: bool,
    fakeroot: bool,
    direct_docker_base: bool,
) -> dict[str, str]:
    task_dir = source_dir / task_id
    environment_dir = task_dir / "environment"
    definition = environment_dir / "container.def"
    output_sif = environment_dir / "container.sif"
    log_path = log_dir / f"{task_id}.log"

    if output_sif.exists() and not force:
        return {"task_id": task_id, "status": "skipped", "path": str(output_sif)}
    if not definition.exists():
        return {
            "task_id": task_id,
            "status": "failed",
            "error": f"missing definition: {definition}",
        }

    definition_text = definition.read_text()
    try:
        portable_definition = make_portable_definition(
            definition_text,
            base_sif,
            direct_docker_base,
        )
    except ValueError as error:
        return {
            "task_id": task_id,
            "status": "failed",
            "error": f"{definition} {error}",
        }

    environment_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    temp_sif_handle = tempfile.NamedTemporaryFile(
        prefix=".container.",
        suffix=".sif",
        dir=environment_dir,
        delete=False,
    )
    temp_sif = Path(temp_sif_handle.name)
    temp_sif_handle.close()
    temp_sif.unlink()
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=".container.",
        suffix=".def",
        dir=environment_dir,
        delete=False,
    ) as handle:
        handle.write(portable_definition)
        temp_definition = Path(handle.name)

    try:
        with log_path.open("w") as log:
            command = [runtime, "build"]
            if fakeroot:
                command.append("--fakeroot")
            command.extend([str(temp_sif), str(temp_definition)])
            result = subprocess.run(
                command,
                cwd=environment_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode:
            temp_sif.unlink(missing_ok=True)
            return {
                "task_id": task_id,
                "status": "failed",
                "error": f"{Path(runtime).name} exited {result.returncode}; see {log_path}",
            }
        temp_sif.replace(output_sif)
        return {"task_id": task_id, "status": "built", "path": str(output_sif)}
    finally:
        temp_definition.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime is None:
        raise SystemExit("neither apptainer nor singularity is available on PATH")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    source_dir = args.source_dir.resolve()
    split_file = args.split_file.resolve()
    base_sif = args.base_sif.resolve()
    log_dir = args.log_dir.resolve()
    if not args.direct_docker_base:
        ensure_base_image(runtime, base_sif, args.pull_base)

    task_ids = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]
    task_ids = task_ids[args.start :]
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    counts = {"built": 0, "skipped": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                build_one,
                runtime,
                source_dir,
                base_sif,
                log_dir,
                task_id,
                args.force,
                args.fakeroot,
                args.direct_docker_base,
            ): task_id
            for task_id in task_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            counts[result["status"]] += 1
            if result["status"] == "failed":
                failures.append(result)
            print(
                f"[{completed}/{len(task_ids)}] {result['task_id']}: "
                f"{result['status']}",
                flush=True,
            )

    summary = {
        "source_dir": str(source_dir),
        "split_file": str(split_file),
        "base_sif": str(base_sif),
        "runtime": runtime,
        "fakeroot": args.fakeroot,
        "direct_docker_base": args.direct_docker_base,
        "requested": len(task_ids),
        "counts": counts,
        "failures": failures,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
