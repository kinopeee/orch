from __future__ import annotations

from pathlib import Path


def run_dir(home: Path, run_id: str) -> Path:
    """Return run directory path."""
    return home / "runs" / run_id


def ensure_run_layout(run_dir: Path) -> None:
    """Ensure all directories required by the run layout exist."""
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
