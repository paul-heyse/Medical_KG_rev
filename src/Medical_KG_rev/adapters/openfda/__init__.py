"""OpenFDA adapters for drug labels, adverse events, and device classifications."""

from .adapter import (
    OpenFDAAdapter,
    OpenFDADeviceAdapter,
    OpenFDADrugEventAdapter,
    OpenFDADrugLabelAdapter,
)

__all__ = [
    "OpenFDAAdapter",
    "OpenFDADrugLabelAdapter",
    "OpenFDADrugEventAdapter",
    "OpenFDADeviceAdapter",
]
