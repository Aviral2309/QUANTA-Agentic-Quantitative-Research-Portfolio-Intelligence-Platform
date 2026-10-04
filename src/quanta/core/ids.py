from datetime import datetime, timezone
import secrets

def new_run_id(prefix: str = "QNT") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}-{secrets.token_hex(3).upper()}"
