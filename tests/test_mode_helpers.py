"""Tests for src.modes.helpers."""

import numpy as np

from src.modes.helpers import apply_pan, nearest_crot, CROT_ANGLES
from src.params import Params, derive


class TestNearestCrot:
    def test_exact_match(self):
        assert nearest_crot(120) == 120

    def test_below_first(self):
        assert nearest_crot(30) == 60

    def test_between(self):
        assert nearest_crot(170) == 180

    def test_above_last(self):
        assert nearest_crot(350) == 300

    def test_midpoint(self):
        # 90 is equidistant from 60 and 120; either is valid
        assert nearest_crot(90) in (60, 120)

    def test_all_angles_reachable(self):
        for a in CROT_ANGLES:
            assert nearest_crot(a) == a


class TestApplyPan:
    def _make_dp(self, pan_amount):
        p = Params(movement=pan_amount)
        return derive(p)

    def test_zero_pan_no_shift(self):
        frame = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        dp = self._make_dp(0.0)
        result = apply_pan(frame, 0.0, dp, 0.7, 0.5, 0.0, 0.0)
        np.testing.assert_array_equal(result, frame)

    def test_nonzero_pan_shifts(self):
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[:, 0, :] = 255  # mark first column
        dp = self._make_dp(1.0)
        result = apply_pan(frame, 1.0, dp, 0.7, 0.5, 100, 0)
        # With high pan_amount and large x_amp, the column should shift
        assert not np.array_equal(result, frame)

    def test_preserves_shape_dtype(self):
        frame = np.random.randint(0, 255, (8, 12, 3), dtype=np.uint8)
        dp = self._make_dp(0.5)
        result = apply_pan(frame, 2.0, dp, 0.4, 0.3, 30, 20)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
