"""Deterministic structured-operator reproduction package."""


def reproduce() -> dict:
    """Run the portable experiment without importing it at package import time."""
    from .experiment import reproduce as _reproduce

    return _reproduce()


__all__ = ["reproduce"]
