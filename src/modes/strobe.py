"""Strobe mode — hard-switching color variants on beats."""

import numpy as np

from src.modes.base import Mode
from src.transforms import blend, fast_zoom


class StrobeMode(Mode):
    def __init__(self, w: int, h: int):
        super().__init__(w, h)
        self._trail: np.ndarray | None = None

    def reset(self, w: int, h: int) -> None:
        super().reset(w, h)
        self._trail = None

    def render(self, t, bass, mids, highs, energy, onset, dp, drift,
               assets, img_blend, frame_key, fps):
        rm = dp.react_mod
        bpm = drift.bpm
        beat_dur = 60.0 / bpm
        beat_phase = (t % beat_dur) / beat_dur
        beat_idx = int(t / beat_dur)

        # Select variant based on beat and audio
        if bass > 0.7 and beat_phase < 0.12:
            variant = ["inverted", "solar_80", "poster_2", "solar_200"][beat_idx % 4]
            key = frame_key(t, variant)
            frame = assets.get(key) if key else None
        elif energy > 0.4:
            variants = ["solar_140", "poster_3", "deep", "crot_180"]
            v1 = variants[beat_idx % len(variants)]
            v2 = variants[(beat_idx + 1) % len(variants)]
            k1 = frame_key(t, v1)
            k2 = frame_key(t, v2)
            f1 = assets.get(k1) if k1 else None
            f2 = assets.get(k2) if k2 else None
            if f1 is not None and f2 is not None:
                frame = blend(f1, f2, beat_phase)
            else:
                frame = f1 or f2
        else:
            frame = img_blend(t, "abstract")

        if frame is None:
            frame = img_blend(t, "abstract")

        result = frame.copy()

        # Zoom punch on beat
        bp = drift.beat_pulse
        if bp > 0.1 and bass > 0.3:
            punch_zoom = 1.0 + bp * bass * 0.18
            result = fast_zoom(result, punch_zoom)

        # Flash on strong onset
        if onset > 0.5:
            flash = int(onset * 160)
            tint_r = int(bass * 40)
            tint_b = int(highs * 40)
            if beat_idx % 2 == 0:
                r = np.minimum(result[:, :, 0].astype(np.int16) + flash + tint_r, 255).astype(np.uint8)
                g = np.minimum(result[:, :, 1].astype(np.int16) + flash, 255).astype(np.uint8)
                b = np.minimum(result[:, :, 2].astype(np.int16) + flash + tint_b, 255).astype(np.uint8)
                result = np.stack([r, g, b], axis=2)
            else:
                result = np.maximum(result.astype(np.int16) - flash, 0).astype(np.uint8)

        # Inversion on onset
        if onset > 0.85:
            result = 255 - result

        # Persistence trails
        if self._trail is not None:
            trail_mix = 0.3 + energy * 0.2
            result = blend(self._trail, result, trail_mix)
        self._trail = result.copy()

        # Moving scan lines
        if highs > 0.2:
            spacing = max(2, 5 - int(energy * 3))
            offset = int(t * (60 + energy * 200 * rm)) % spacing
            result[offset::spacing] = (result[offset::spacing].astype(np.float32) * 0.5).astype(np.uint8)

        # Pan jitter on beat
        if beat_phase < 0.1 and onset > 0.3:
            jitter = int((np.random.random() - 0.5) * onset * 40 * rm)
            result = np.roll(result, jitter, axis=1)

        return result
