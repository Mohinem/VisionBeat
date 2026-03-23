"""Math helpers used by gesture detection."""

from __future__ import annotations

import math
from collections.abc import Iterable


def l1_velocity(delta_components: Iterable[float], elapsed: float) -> float:
    """Compute a simple motion velocity estimate from vector components."""
    values = list(delta_components)
    safe_elapsed = max(elapsed, 1e-6)
    try:
        import numpy as np

        return float(np.linalg.norm(np.asarray(values, dtype=float), ord=1) / safe_elapsed)
    except ModuleNotFoundError:
        return math.fsum(abs(value) for value in values) / safe_elapsed
