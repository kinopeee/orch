from __future__ import annotations

import stat
from pathlib import Path


def is_symlink_path(path: Path, *, fail_closed: bool = True) -> bool:
    try:
        return path.is_symlink()
    except FileNotFoundError:
        return False
    except (OSError, RuntimeError):
        return fail_closed


def has_symlink_ancestor(path: Path) -> bool:
    current = path.parent
    while True:
        try:
            meta = current.lstat()
        except FileNotFoundError:
            pass
        except (OSError, RuntimeError):
            return True
        else:
            if stat.S_ISLNK(meta.st_mode):
                return True
        if current == current.parent:
            return False
        current = current.parent
