"""Radial mode — concentric pulsating color waves."""

import numpy as np

from src.modes.base import Mode
from src.modes.helpers import apply_pan, nearest_crot
from src.transforms import blend, fast_zoom


class RadialMode(Mode):
    def __init__(self, w: int, h: int):
        super().__init__(w, h)
        self._angle = 0.0

    def reset(self, w: int, h: int) -> None:
        super().reset(w, h)
        self._angle = 0.0

    def render(self, t, bass, mids, highs, energy, onset, dp, drift,
               assets, img_blend, frame_key, fps):
        rm = dp.react_mod

        # Crossfade between radial variants
        k_lo = frame_key(t, "radial_30")
        lo = assets.get(k_lo) if k_lo else None
        k_mid = frame_key(t, "radial_60")
        mid_f = assets.get(k_mid) if k_mid else None
        k_hi = frame_key(t, "radial_90")
        hi = assets.get(k_hi) if k_hi else None

        if lo is None:
            return img_blend(t, "abstract")

        if bass < 0.4:
            frame = blend(lo, mid_f, bass * 2.5) if mid_f is not None else lo
        else:
            frame = blend(mid_f, hi, (bass - 0.4) * 1.67) if mid_f is not None and hi is not None else (mid_f if mid_f is not None else lo)

        # Breathing zoom + beat punch
        bp = drift.beat_pulse
        zoom = 1.0 + bass * 0.12 + np.sin(t * 0.8) * 0.03 + bp * bass * 0.1
        if zoom > 1.01:
            frame = fast_zoom(frame, zoom)

        # Rotation — accelerates on beats
        self._angle += (0.3 + mids * 2.0 + bass * 1.5 + bp * 3.0) * dp.rotation_speed * rm / fps
        if dp.rotation_speed > 0.01:
            rx = int(np.sin(self._angle) * self.w * 0.08)
            ry = int(np.cos(self._angle) * self.h * 0.08)
            if rx != 0:
                frame = np.roll(frame, rx, axis=1)
            if ry != 0:
                frame = np.roll(frame, ry, axis=0)

        # Deep abstract on low energy
        if energy < 0.3:
            k_deep = frame_key(t, "deep")
            deep = assets.get(k_deep) if k_deep else None
            if deep is not None:
                frame = blend(frame, deep, (0.3 - energy) * 2.5)

        # Color rotation
        closest = nearest_crot((t * 25 + mids * 100) % 360)
        k_crot = frame_key(t, f"crot_{closest}")
        crot = assets.get(k_crot) if k_crot else None
        if crot is not None:
            mix = 0.15 + mids * 0.35 + energy * 0.1
            frame = blend(frame, crot, min(mix, 0.6))

        # Pan drift
        frame = apply_pan(frame, t, dp, 0.4, 0.3, 30 * energy, 20 * mids)

        return frame
