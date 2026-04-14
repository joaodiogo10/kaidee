"""Tests for visual mode rendering."""

import numpy as np
import pytest

from src.modes import MODE_CLASSES
from src.modes.feedback import FeedbackMode
from src.modes.strobe import StrobeMode
from src.modes.kaleidoscope import KaleidoscopeMode
from src.params import Params, derive
from src.drift import DriftValues

W, H = 80, 64


class MockAssets:
    def __init__(self, w, h):
        self._frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def get(self, key):
        return self._frame if key else None


def _make_dp():
    return derive(Params())


def _make_drift():
    return DriftValues(bpm=128.0, beat_pulse=0.5, bar_pulse=0.3,
                       beat_phase=0.2, bar_phase=0.1, bar_idx=4)


def _mock_img_blend(t, variant):
    return np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)


def _mock_frame_key(t, variant):
    return f"img0_{variant}"


class TestAllModes:
    @pytest.mark.parametrize("name,cls", list(MODE_CLASSES.items()))
    def test_returns_valid_frame(self, name, cls):
        mode = cls(W, H)
        result = mode.render(
            t=1.0, bass=0.5, mids=0.4, highs=0.3,
            energy=0.4, onset=0.2, dp=_make_dp(), drift=_make_drift(),
            assets=MockAssets(W, H), img_blend=_mock_img_blend,
            frame_key=_mock_frame_key, fps=60,
        )
        assert result.shape == (H, W, 3)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("name,cls", list(MODE_CLASSES.items()))
    def test_silent_audio(self, name, cls):
        """Mode should not crash with zero audio values."""
        mode = cls(W, H)
        result = mode.render(
            t=0.5, bass=0.0, mids=0.0, highs=0.0,
            energy=0.0, onset=0.0, dp=_make_dp(), drift=_make_drift(),
            assets=MockAssets(W, H), img_blend=_mock_img_blend,
            frame_key=_mock_frame_key, fps=60,
        )
        assert result.shape == (H, W, 3)

    @pytest.mark.parametrize("name,cls", list(MODE_CLASSES.items()))
    def test_max_audio(self, name, cls):
        """Mode should not crash with max audio values."""
        mode = cls(W, H)
        result = mode.render(
            t=2.0, bass=1.0, mids=1.0, highs=1.0,
            energy=1.0, onset=1.0, dp=_make_dp(), drift=_make_drift(),
            assets=MockAssets(W, H), img_blend=_mock_img_blend,
            frame_key=_mock_frame_key, fps=60,
        )
        assert result.shape == (H, W, 3)


class TestFeedbackMode:
    def test_accumulates_buffer(self):
        mode = FeedbackMode(W, H)
        assert mode.buf is None
        for i in range(3):
            mode.render(
                t=i * 0.1, bass=0.5, mids=0.3, highs=0.2,
                energy=0.4, onset=0.1, dp=_make_dp(), drift=_make_drift(),
                assets=MockAssets(W, H), img_blend=_mock_img_blend,
                frame_key=_mock_frame_key, fps=60,
            )
        assert mode.buf is not None
        assert mode.buf.shape == (H, W, 3)


class TestStrobeMode:
    def test_trail_persistence(self):
        mode = StrobeMode(W, H)
        assert mode._trail is None
        for i in range(2):
            mode.render(
                t=i * 0.5, bass=0.5, mids=0.3, highs=0.2,
                energy=0.4, onset=0.1, dp=_make_dp(), drift=_make_drift(),
                assets=MockAssets(W, H), img_blend=_mock_img_blend,
                frame_key=_mock_frame_key, fps=60,
            )
        assert mode._trail is not None


class TestKaleidoscopeMode:
    def test_grids_match_dimensions(self):
        mode = KaleidoscopeMode(W, H)
        assert mode._base_radius.shape == (H, W)
        assert mode._base_theta.shape == (H, W)

    def test_reset_rebuilds_grids(self):
        mode = KaleidoscopeMode(W, H)
        mode.reset(100, 50)
        assert mode._base_radius.shape == (50, 100)
        assert mode.w == 100
        assert mode.h == 50


class TestModeReset:
    @pytest.mark.parametrize("name,cls", list(MODE_CLASSES.items()))
    def test_reset_updates_dimensions(self, name, cls):
        mode = cls(W, H)
        mode.reset(120, 90)
        assert mode.w == 120
        assert mode.h == 90
