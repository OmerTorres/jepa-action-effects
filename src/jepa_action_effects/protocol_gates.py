"""Corrected manipulation-check rules for future sealed protocols (D026 findings 2, 7).

- **Per-cell readout gate.** SHIFT-002 pooled the real-frame slope over doses within
  each state × structure cell, so opposite dose-dependent readout biases could cancel.
  This gate requires every cell used in a contrast to pass on its own.
- **Two-sided faithfulness.** The SHIFT rules accepted any CI whose lower bound was
  above 0.9, so [2, 3] passed, and any wide CI containing 1 passed too. This rule
  requires the whole CI to lie within 1 ± tolerance and to be narrow enough to be
  informative.

These do not change D023 or any completed verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

READOUT_BAND = (0.85, 1.15)
FAITHFUL_TOLERANCE = 0.1
FAITHFUL_MAX_WIDTH = 0.2


def readout_gate_per_cell(
    intervals: Mapping[str, Sequence[float] | None], band: tuple[float, float] = READOUT_BAND
) -> dict[str, Any]:
    """Every contrast cell's slope_real CI must lie strictly inside `band`."""
    failing = {
        cell: interval
        for cell, interval in intervals.items()
        if interval is None or not (band[0] < interval[0] and interval[1] < band[1])
    }
    return {"passed": bool(intervals) and not failing, "failing_cells": failing}


def faithful(
    interval: Sequence[float] | None,
    tolerance: float = FAITHFUL_TOLERANCE,
    max_width: float = FAITHFUL_MAX_WIDTH,
) -> bool:
    """CI inside [1 − tolerance, 1 + tolerance] and no wider than `max_width`."""
    if interval is None:
        return False
    low, high = interval
    return 1 - tolerance <= low and high <= 1 + tolerance and high - low <= max_width
