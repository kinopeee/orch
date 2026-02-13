from __future__ import annotations

import json
import os
from pathlib import Path

from orch.state.model import RunState
from orch.util.errors import StateError


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def load_state(run_dir: Path) -> RunState:
    state_path = run_dir / "state.json"
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StateError(f"state file not found: {state_path}") from exc
    except UnicodeError as exc:
        raise StateError(f"failed to decode state file as utf-8: {state_path}") from exc
    except OSError as exc:
        raise StateError(f"failed to read state file: {state_path}") from exc
    except json.JSONDecodeError as exc:
        raise StateError(f"invalid state json: {state_path}") from exc
    if not isinstance(raw, dict):
        raise StateError("state root must be object")
    return RunState.from_dict(raw)


def save_state_atomic(run_dir: Path, state: RunState) -> None:
    state_path = run_dir / "state.json"
    tmp_path = run_dir / "state.json.tmp"
    payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(payload + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, state_path)
    _fsync_directory(run_dir)
