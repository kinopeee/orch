from __future__ import annotations

import stat
from pathlib import Path

from orch.util.path_guard import has_symlink_ancestor, is_symlink_path


def run_dir(home: Path, run_id: str) -> Path:
    """Return run directory path."""
    return home / "runs" / run_id


def _ensure_directory(path: Path, *, parents: bool = False) -> None:
    if has_symlink_ancestor(path):
        raise OSError(f"path must not include symlink: {path}")
    if is_symlink_path(path):
        raise OSError(f"path must not be symlink: {path}")
    try:
        path.mkdir(parents=parents, exist_ok=True)
    except (OSError, RuntimeError) as exc:
        raise OSError(f"failed to create directory path: {path}") from exc
    try:
        meta = path.lstat()
    except (OSError, RuntimeError) as exc:
        raise OSError(f"path must be directory: {path}") from exc
    if is_symlink_path(path) or not stat.S_ISDIR(meta.st_mode):
        raise OSError(f"path must be directory: {path}")


def ensure_run_layout(run_dir: Path) -> None:
    """Ensure all directories required by the run layout exist."""
    _ensure_directory(run_dir, parents=True)
    _ensure_directory(run_dir / "logs")
    _ensure_directory(run_dir / "artifacts")
    _ensure_directory(run_dir / "report")
