from __future__ import annotations

from pathlib import Path

from orch.util.path_guard import has_symlink_ancestor


def run_dir(home: Path, run_id: str) -> Path:
    """Return run directory path."""
    return home / "runs" / run_id


def _ensure_directory(path: Path, *, parents: bool = False) -> None:
    if has_symlink_ancestor(path):
        raise OSError(f"path contains symlink component: {path}")
    if path.is_symlink():
        raise OSError(f"path must not be symlink: {path}")
    path.mkdir(parents=parents, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise OSError(f"path must be directory: {path}")


def ensure_run_layout(run_dir: Path) -> None:
    """Ensure all directories required by the run layout exist."""
    _ensure_directory(run_dir, parents=True)
    _ensure_directory(run_dir / "logs")
    _ensure_directory(run_dir / "artifacts")
    _ensure_directory(run_dir / "report")
