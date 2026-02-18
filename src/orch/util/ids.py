from __future__ import annotations

import secrets
from datetime import datetime


def new_run_id(now: datetime) -> str:
    """Create run id in YYYYMMDD_HHMMSS_xxxxxx format."""
    return f"{now:%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"
