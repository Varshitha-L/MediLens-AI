"""Shared constants for the MediLens AI module contracts."""

ALLOWED_STATUSES = {"success", "partial", "error"}
ALLOWED_FLAGS = {"high", "low", "within_range", "unknown"}
SAFETY_MESSAGE = (
    "MediLens AI supports health awareness and does not replace professional medical advice."
)


def status_is_valid(value: object) -> bool:
    """Return whether a module result uses one of the agreed status strings."""
    return isinstance(value, str) and value in ALLOWED_STATUSES
