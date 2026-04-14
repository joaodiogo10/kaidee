"""Feedback mode — recursive zoom tunnel."""

import numpy as np

from src.modes.base import Mode
from src.modes.helpers import apply_pan, nearest_crot
from src.transforms import blend, fast_zoom


class FeedbackMode(Mode):
    def __init__(self, w: int, h: int):
        super().__init__(w, h)
        self.buf: np.ndarray | None = None

    def reset(self, w: int, h: int) -> None:
        super().reset(w, h)
        self.buf = None

    def render(self, t, bass, mids, highs, energy, onset, dp, drift,
               assets, img_blend, frame_key, fps):
        rm = dp.react_mod

        if self.buf is None:
            key = frame_key(t, "abstract")
            src = assets.get(key) if key else None
            self.buf = src.astype(np.float32) if src is not None else \
                np.zeros((self.h, self.w, 3), np.float32)

        # Zoom driven by bass + beat pulse
        bp = drift.beat_pulse
        zoom = 1.01 + bass * 0.06 + energy * 0.02 + bp * bass * 0.08
        buf_u8 = self.buf.clip(0, 255).astype(np.uint8)
        zoomed = fast_zoom(buf_u8, zoom).astype(np.float32)

        # Pan
        zoomed = apply_pan(zoomed, t, dp, 0.7, 0.5, mids * 25, highs * 15)

        # Rotation
        if energy > 0.1 and dp.rotation_speed > 0.01:
            angle_rad = np.sin(t * 0.3) * energy * 5 * dp.rotation_speed * rm * 0.02
            rx = int(np.sin(angle_rad) * self.w * 0.1)
            ry = int(np.cos(angle_rad) * self.h * 0.1)
            if rx != 0:
                zoomed = np.roll(zoomed, rx, axis=1)
            if ry != 0:
                zoomed = np.roll(zoomed, ry, axis=0)

        # Decay
        decay = 0.90 + energy * 0.06
        zoomed *= decay

        # Color source
        closest = nearest_crot((t * 30 + mids * 60) % 360)
        key = frame_key(t, f"crot_{closest}")
        source = assets.get(key) if key else None
        if source is None:
            key = frame_key(t, "abstract")
            source = assets.get(key) if key else None

        # Source injection — stronger on beats
        alpha = 0.04 + bass * 0.12 + onset * 0.25 + energy * 0.05 + bp * 0.15
        if source is not None:
            self.buf = zoomed * (1 - alpha) + source.astype(np.float32) * alpha
        else:
            self.buf = zoomed

        # Channel swap on onset
        if onset > 0.6:
            self.buf[:, :, 0], self.buf[:, :, 1], self.buf[:, :, 2] = (
                self.buf[:, :, 1].copy(),
                self.buf[:, :, 2].copy(),
                self.buf[:, :, 0].copy(),
            )

        # Color cycling — apply in float32 before final clip+convert
        hue_shift = np.sin(t * 0.2) * 20
        if abs(hue_shift) > 2:
            self.buf[:, :, 0] += hue_shift
            self.buf[:, :, 1] -= hue_shift * 0.5
            self.buf[:, :, 2] += hue_shift / 3

        return self.buf.clip(0, 255).astype(np.uint8)
