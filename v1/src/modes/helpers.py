"""Shared helpers for visual modes."""

import numpy as np

from src.params import DerivedParams

CROT_ANGLES = (60, 120, 180, 240, 300)


def apply_pan(frame: np.ndarray, t: float, dp: DerivedParams,
              x_freq: float, y_freq: float,
              x_amp: float, y_amp: float) -> np.ndarray:
    """Apply oscillating pan offset based on dp.pan_amount."""
    pan_scale = 0.5 + dp.pan_amount * 12.0
    px = int(np.sin(t * x_freq) * x_amp * pan_scale)
    py = int(np.cos(t * y_freq) * y_amp * pan_scale)
    if px != 0:
        frame = np.roll(frame, px, axis=1)
    if py != 0:
        frame = np.roll(frame, py, axis=0)
    return frame


def nearest_crot(angle_continuous: float) -> int:
    """Return the nearest pre-compiled color rotation angle."""
    return min(CROT_ANGLES, key=lambda a: abs(a - angle_continuous))
