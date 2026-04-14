"""Kaleidoscope mode — mandala/sacred geometry from sharp images."""

import numpy as np
from scipy.ndimage import map_coordinates

from src.modes.base import Mode
from src.modes.helpers import nearest_crot
from src.transforms import blend


class KaleidoscopeMode(Mode):
    uses_sharp = True

    def __init__(self, w: int, h: int):
        super().__init__(w, h)
        self._angle = 0.0
        self._build_grids()

    def _build_grids(self) -> None:
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        cy, cx = self.h / 2.0, self.w / 2.0
        self._base_radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        self._base_theta = np.arctan2(yy - cy, xx - cx)
        self._cy, self._cx = cy, cx

    def reset(self, w: int, h: int) -> None:
        super().reset(w, h)
        self._angle = 0.0
        self._build_grids()

    def render(self, t, bass, mids, highs, energy, onset, dp, drift,
               assets, img_blend, frame_key, fps):
        rm = dp.react_mod

        img_sharp = img_blend(t, "sharp")
        closest = nearest_crot((t * 30 + bass * 60) % 360)
        img_crot = img_blend(t, f"crot_{closest}")

        if img_sharp is None:
            return np.zeros((self.h, self.w, 3), np.uint8)

        if img_crot is not None:
            img = blend(img_sharp, img_crot, 0.2 + mids * 0.3)
        else:
            img = img_sharp

        segments = max(3, min(12, 4 + int(highs * 4 * rm) + int(bass * 2 * rm)))
        wedge = 2 * np.pi / segments

        bp = drift.beat_pulse
        self._angle += (0.5 + bass * 4.0 + mids * 1.5 + bp * 5.0) * dp.rotation_speed * rm / fps
        wobble = np.sin(t * 1.5) * mids * 0.3 * rm + bp * 0.15
        zoom = 1.0 + mids * 0.3 + bass * 0.15 + np.sin(t * 0.7) * 0.05 + bp * bass * 0.12

        cx = self._cx + np.sin(t * 0.4) * 30 * mids * rm
        cy = self._cy + np.cos(t * 0.3) * 20 * bass * rm

        theta = self._base_theta - self._angle + wobble
        radius = self._base_radius / zoom

        th_mod = theta % (2 * np.pi)
        wp = th_mod % wedge
        wi = (th_mod // wedge).astype(np.int32)
        mirror = (wi & 1) == 1
        wp[mirror] = wedge - wp[mirror]

        src_y = np.clip(cy + radius * np.sin(wp), 0, self.h - 1).astype(np.intp)
        src_x = np.clip(cx + radius * np.cos(wp), 0, self.w - 1).astype(np.intp)

        return img[src_y, src_x]
