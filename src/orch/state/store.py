"""State file persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orch.state.model import RunState


def _state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def load_state(run_dir: Path) -> RunState:
    """Load state.json for run."""
    path = _state_path(run_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("state.json root must be object")
    data: dict[str, Any] = raw
    return RunState.from_dict(data)


def save_state_atomic(run_dir: Path, state: RunState) -> None:
    """Atomically save state json via temp file + replace."""
    path = _state_path(run_dir)
    tmp = run_dir / "state.json.tmp"
    payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=False)
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(path)
