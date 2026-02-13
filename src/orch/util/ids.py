"""ID generation utilities."""

from datetime import datetime
from secrets import token_hex


def new_run_id(now: datetime) -> str:
    """Create run id: YYYYMMDD_HHMMSS_<6chars>."""
    ts = now.strftime("%Y%m%d_%H%M%S")
    suffix = token_hex(3)
    return f"{ts}_{suffix}"
