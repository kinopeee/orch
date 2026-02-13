from __future__ import annotations

from pathlib import Path


def cancel_requested(run_dir: Path) -> bool:
    return (run_dir / "cancel.request").exists()


def write_cancel_request(run_dir: Path) -> None:
    path = run_dir / "cancel.request"
    path.write_text("cancel requested\n", encoding="utf-8")
