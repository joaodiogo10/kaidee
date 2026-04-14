"""Glitch mode — pixel-sorted color streams + band displacement."""

import numpy as np

from src.modes.base import Mode
from src.transforms import blend, fast_zoom


class GlitchMode(Mode):
    def render(self, t, bass, mids, highs, energy, onset, dp, drift,
               assets, img_blend, frame_key, fps):
        rm = dp.react_mod

        abstract = img_blend(t, "abstract")
        sorted_f = img_blend(t, "sorted")
        inverted = img_blend(t, "inverted")
        if abstract is None:
            return np.zeros((self.h, self.w, 3), np.uint8)

        # 3-way blend
        frame = blend(abstract, sorted_f, bass * 0.8) if sorted_f is not None else abstract
        if inverted is not None and highs > 0.2:
            frame = blend(frame, inverted, highs * 0.4)

        result = frame.copy()

        # Zoom pulse on beats + bass
        bp = drift.beat_pulse
        zoom = 1.0 + bp * bass * 0.12 + max(0, bass - 0.3) * 0.05
        if zoom > 1.01:
            result = fast_zoom(result, zoom)

        # Evolving band displacement — cap bands for performance
        num_bands = min(4 + int(mids * 6), 8)
        band_w = max(1, self.w // num_bands)
        for b in range(num_bands):
            x0 = b * band_w
            x1 = min((b + 1) * band_w, self.w)
            phase = t * 3 + b * 1.7 + bass * 5
            shift = int(np.sin(phase) * (20 + mids * 50 + onset * 40) * rm)
            if abs(shift) > 1:
                result[:, x0:x1] = np.roll(result[:, x0:x1], shift, axis=0)
            h_shift = int(np.cos(phase * 0.7) * highs * 20 * rm)
            if abs(h_shift) > 1:
                result[:, x0:x1] = np.roll(result[:, x0:x1], h_shift, axis=1)

        # Scanning glitch bar
        scan_y = int((t * 120 + bass * 200 * rm) % self.h)
        scan_h = max(4, int((bass + mids) * 20 * rm))
        y0, y1 = scan_y, min(scan_y + scan_h, self.h)
        if y1 > y0:
            result[y0:y1] = 255 - result[y0:y1]

        # RGB channel separation — intensified on beats
        sep = int((highs * 30 + bass * 15 + bp * 20) * (1 + 0.5 * np.sin(t * 2)) * rm)
        if sep > 2:
            result[:, :, 0] = np.roll(result[:, :, 0], sep, axis=1)
            result[:, :, 2] = np.roll(result[:, :, 2], -sep, axis=1)
            g_shift = int(mids * 8 * np.sin(t * 1.5) * rm)
            if g_shift != 0:
                result[:, :, 1] = np.roll(result[:, :, 1], g_shift, axis=0)

        # Mix solarized on strong bass
        if bass > 0.4:
            key = frame_key(t, "solar_80")
            solar = assets.get(key) if key else None
            if solar is not None:
                result = blend(result, solar, min((bass - 0.4) * 1.2, 0.7))

        return result
