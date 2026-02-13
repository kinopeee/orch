from __future__ import annotations

from pathlib import Path


def cancel_requested(run_dir: Path) -> bool:
    path = run_dir / "cancel.request"
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def write_cancel_request(run_dir: Path) -> None:
    path = run_dir / "cancel.request"
    if path.is_symlink():
        raise OSError("cancel request path must not be symlink")
    if path.exists() and not path.is_file():
        raise OSError("cancel request path must be regular file")
    path.write_text("cancel requested\n", encoding="utf-8")


def clear_cancel_request(run_dir: Path) -> None:
    path = run_dir / "cancel.request"
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
    except OSError:
        return
