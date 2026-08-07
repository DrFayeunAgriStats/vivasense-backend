from fastapi import HTTPException, Request

_PILOT_GATE_MESSAGE = "This module is still being validated and isn't part of the current pilot."


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def require_genetics_admin(request: Request) -> None:
    # Backend pilot gate (separate from Pro/licensing flags).
    # Caller must forward an explicit admin signal.
    admin_header = request.headers.get("X-VivaSense-Is-Admin", "")
    if not _is_truthy(admin_header):
        raise HTTPException(
            status_code=403,
            detail=_PILOT_GATE_MESSAGE,
        )
