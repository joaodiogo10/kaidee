"""Displacement mode — fluid flowing warp (paint-in-water)."""

import numpy as np

from src.modes.base import Mode
from src.modes.helpers import apply_pan, nearest_crot
from src.transforms import blend, fast_zoom


class DisplacementMode(Mode):
    def __init__(self, w: int, h: int):
        super().__init__(w, h)
        self._angle = 0.0

    def reset(self, w: int, h: int) -> None:
        super().reset(w, h)
        self._angle = 0.0

    def render(self, t, bass, mids, highs, energy, onset, dp, drift,
               assets, img_blend, frame_key, fps):
        rm = dp.react_mod

        k_lo = frame_key(t, "fluid_40")
        lo = assets.get(k_lo) if k_lo else None
        k_mid = frame_key(t, "fluid_80")
        mid_f = assets.get(k_mid) if k_mid else None
        k_hi = frame_key(t, "fluid_120")
        hi = assets.get(k_hi) if k_hi else None

        if lo is None:
            return img_blend(t, "abstract")

        if bass < 0.3:
            frame = blend(lo, mid_f, bass * 3.3) if mid_f is not None else lo
        else:
            frame = blend(mid_f, hi, (bass - 0.3) * 1.43) if mid_f is not None and hi is not None else (mid_f if mid_f is not None else lo)

        # Breathing zoom + beat punch
        bp = drift.beat_pulse
        zoom = 1.0 + np.sin(t * 0.6) * 0.04 + bass * 0.08 + bp * bass * 0.1
        if zoom > 1.01:
            frame = fast_zoom(frame, zoom)

        # Liquid pan (audio-modulated phase, can't use apply_pan helper)
        pm = dp.pan_amount
        pan_scale = 0.5 + pm * 12.0
        pan_x = int(np.sin(t * 0.5 + mids * 3) * (15 + energy * 30) * pan_scale)
        pan_y = int(np.cos(t * 0.35 + bass * 2) * (10 + energy * 20) * pan_scale)
        if pan_x != 0:
            frame = np.roll(frame, pan_x, axis=1)
        if pan_y != 0:
            frame = np.roll(frame, pan_y, axis=0)

        # Wave distortion — intensified on beats
        if mids > 0.2:
            wave_amp = (mids * 20 + energy * 10 + bp * 25) * rm
            wave_freq = 0.02 + highs * 0.03
            y_all = np.arange(self.h)
            shifts = (np.sin(y_all * wave_freq + t * 3) * wave_amp).astype(np.int32)
            x_base = np.arange(self.w)
            shifted_x = (x_base[np.newaxis, :] - shifts[:, np.newaxis]) % self.w
            frame = frame[y_all[:, np.newaxis], shifted_x]

        # Color rotation
        closest = nearest_crot((t * 20 + mids * 80) % 360)
        k_crot = frame_key(t, f"crot_{closest}")
        crot = assets.get(k_crot) if k_crot else None
        if crot is not None:
            mix = 0.15 + mids * 0.3 + energy * 0.15
            frame = blend(frame, crot, min(mix, 0.6))

        # Rotation wobble
        self._angle += (0.2 + energy * 1.0) * dp.rotation_speed * rm / fps
        if dp.rotation_speed > 0.01:
            wobble = np.sin(self._angle) * 12
            rx = int(np.sin(np.radians(wobble)) * self.w * 0.08)
            ry = int(np.cos(np.radians(wobble)) * self.h * 0.05)
            if rx != 0:
                frame = np.roll(frame, rx, axis=1)
            if ry != 0:
                frame = np.roll(frame, ry, axis=0)

        return frame
