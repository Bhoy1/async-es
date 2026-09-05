#!/usr/bin/env python3
"""Apply the unprivileged-cluster runtime patch to the official repository."""

from __future__ import annotations

import argparse
from pathlib import Path


INSTANCE_START_ORIGINAL = '''        start_cmd = [
            "apptainer", "instance", "start",
            "--containall",
            "--writable-tmpfs",
            "--bind", f"{self.temp_dir}:{self.temp_dir}",
            "--cleanenv",
            str(self.sif_path),
            self.instance_name,
        ]
'''
INSTANCE_START_LEGACY_PATCHED = '''        start_cmd = [
            "apptainer", "instance", "start",
            "--fakeroot",
            "--userns",
            "--containall",
            "--writable-tmpfs",
            "--bind", f"{self.temp_dir}:{self.temp_dir}",
            "--cleanenv",
            str(self.sif_path),
            self.instance_name,
        ]
'''
INSTANCE_START_PATCHED = '''        start_cmd = ["apptainer", "instance", "start"]
        if os.environ.get("ENDLESS_APPTAINER_FAKEROOT", "1") != "0":
            start_cmd.extend(["--fakeroot", "--userns"])
        else:
            start_cmd.append("--userns")
        start_cmd.extend([
            "--containall",
            "--writable-tmpfs",
            "--bind", f"{self.temp_dir}:{self.temp_dir}",
            "--cleanenv",
            str(self.sif_path),
            self.instance_name,
        ])
'''

SHELL_INIT_ORIGINAL = '''            init_script = (
                "set -o pipefail 2>/dev/null; "
                "export PS1='[$PWD]$ '; "
'''
SHELL_INIT_PATCHED = '''            init_script = (
                "set -o pipefail 2>/dev/null; "
                "ulimit -f 1048576 2>/dev/null || true; "
                "export PS1='[$PWD]$ '; "
'''

TIMEOUT_ORIGINAL = '''        # Handle timeout
        if code is None:
            if self.verbose:
                print(f"⚠️  Command timed out after {timeout or self.read_timeout}s")
            return False, f"Command timed out. Partial output:\\n{raw_out[:500]}"
'''
TIMEOUT_PATCHED = '''        # Handle timeout and ensure the foreground process cannot continue running.
        if code is None:
            timeout_seconds = timeout or self.read_timeout
            if self.verbose:
                print(f"⚠️  Command timed out after {timeout_seconds}s; interrupting it")
            interrupted_out = ""
            interrupted_code = None
            try:
                os.write(self.master_fd, b"\\x03")
                interrupted_out, interrupted_code = self._read_until_marker(timeout=3.0)
            except Exception:
                interrupted_code = None
            raw_out += interrupted_out
            if interrupted_code is None:
                self._stop_shell()
                self._stop_instance()
                return False, (
                    "Command timed out and sandbox was terminated. Partial output:\\n"
                    f"{raw_out[:500]}"
                )
            return False, (
                "Command timed out and was interrupted. Partial output:\\n"
                f"{raw_out[:500]}"
            )
'''

MISSING_INSTANCE_ORIGINAL = '''        if self.shell_process:
            return True

        # Create PTY pair
'''
MISSING_INSTANCE_PATCHED = '''        if self.shell_process:
            return True
        if not self.instance_name:
            if self.verbose:
                print("Cannot start shell without a live Apptainer instance")
            return False

        # Create PTY pair
'''

EARLY_SHELL_EXIT_ORIGINAL = '''                if leftover:
                    print("Shell start output:\\n", leftover)
                return False
'''
EARLY_SHELL_EXIT_PATCHED = '''                if leftover:
                    print("Shell start output:\\n", leftover)
                self._stop_shell()
                return False
'''

SHELL_INIT_TIMEOUT_ORIGINAL = '''            if code is None:
                if self.verbose:
                    print("Shell init timed out.")
                return False
'''
SHELL_INIT_TIMEOUT_PATCHED = '''            if code is None:
                if self.verbose:
                    print("Shell init timed out.")
                self._stop_shell()
                return False
'''

SHELL_START_EXCEPTION_ORIGINAL = '''        except Exception as e:
            if self.verbose:
                print(f"Failed to start shell: {e}")
            return False
'''
SHELL_START_EXCEPTION_PATCHED = '''        except Exception as e:
            if self.verbose:
                print(f"Failed to start shell: {e}")
            self._stop_shell()
            return False
'''

DEAD_SHELL_ORIGINAL = '''            if self.verbose:
                print(f"⚠️  Shell process died (exit code: {self.shell_process.returncode}), restarting...")
            self.shell_process = None
            if not self._start_shell():
'''
DEAD_SHELL_PATCHED = '''            if self.verbose:
                print(f"⚠️  Shell process died (exit code: {self.shell_process.returncode}), restarting...")
            self._stop_shell()
            if not self._start_shell():
'''


def patch_text(text: str) -> tuple[str, bool]:
    changed = False
    if INSTANCE_START_PATCHED not in text:
        instance_start = (
            INSTANCE_START_LEGACY_PATCHED
            if INSTANCE_START_LEGACY_PATCHED in text
            else INSTANCE_START_ORIGINAL
        )
        if instance_start not in text:
            raise ValueError("official instance-start block does not match expected source")
        text = text.replace(instance_start, INSTANCE_START_PATCHED, 1)
        changed = True
    replacements = (
        (
            "shell file limit",
            SHELL_INIT_ORIGINAL,
            SHELL_INIT_PATCHED,
        ),
        (
            "command timeout cleanup",
            TIMEOUT_ORIGINAL,
            TIMEOUT_PATCHED,
        ),
        (
            "missing-instance shell guard",
            MISSING_INSTANCE_ORIGINAL,
            MISSING_INSTANCE_PATCHED,
        ),
        (
            "early shell exit cleanup",
            EARLY_SHELL_EXIT_ORIGINAL,
            EARLY_SHELL_EXIT_PATCHED,
        ),
        (
            "shell initialization timeout cleanup",
            SHELL_INIT_TIMEOUT_ORIGINAL,
            SHELL_INIT_TIMEOUT_PATCHED,
        ),
        (
            "shell startup exception cleanup",
            SHELL_START_EXCEPTION_ORIGINAL,
            SHELL_START_EXCEPTION_PATCHED,
        ),
        (
            "dead shell cleanup",
            DEAD_SHELL_ORIGINAL,
            DEAD_SHELL_PATCHED,
        ),
    )
    for name, original, patched in replacements:
        if patched in text:
            continue
        if original not in text:
            raise ValueError(f"official {name} block does not match expected source")
        text = text.replace(original, patched, 1)
        changed = True
    return text, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    args = parser.parse_args()

    env_path = args.official_repo.resolve() / "generator/env.py"
    original = env_path.read_text()
    patched, changed = patch_text(original)
    if changed:
        env_path.write_text(patched)
        print(f"patched {env_path}")
    else:
        print(f"already patched: {env_path}")


if __name__ == "__main__":
    main()
