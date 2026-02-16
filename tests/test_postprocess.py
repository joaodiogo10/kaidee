"""Tests for src.postprocess."""

from src.postprocess import compute_postfx, PostFX
from src.params import Params, derive
from src.drift import DriftValues


def _default_args(**overrides):
    """Build default arguments for compute_postfx."""
    args = dict(
        t=1.0, fps=60.0, dp=derive(Params()), drift=DriftValues(),
        bass=0.5, energy=0.4, params=Params(), global_rot_angle=0.0,
    )
    args.update(overrides)
    return args


class TestComputePostFX:
    def test_returns_postfx_and_angle(self):
        fx, angle = compute_postfx(**_default_args())
        assert isinstance(fx, PostFX)
        assert isinstance(angle, float)

    def test_default_brightness_neutral(self):
        """brightness=0.5 -> brightness multiplier ~1.0"""
        fx, _ = compute_postfx(**_default_args(params=Params(brightness=0.5)))
        assert abs(fx.brightness - 1.0) < 0.01

    def test_high_brightness(self):
        fx, _ = compute_postfx(**_default_args(params=Params(brightness=1.0)))
        assert fx.brightness > 2.0

    def test_low_brightness(self):
        fx, _ = compute_postfx(**_default_args(params=Params(brightness=0.0)))
        assert fx.brightness < 0.5

    def test_zoom_default_neutral(self):
        """zoom=0.33 (neutral) -> zoom ~1.0"""
        fx, _ = compute_postfx(**_default_args(params=Params(zoom=0.33)))
        assert 0.9 < fx.zoom < 1.1

    def test_zoom_manual_in(self):
        fx, _ = compute_postfx(**_default_args(params=Params(zoom=1.0)))
        assert fx.zoom > 1.5

    def test_trail_off_at_low_perception(self):
        """Low perception -> trail=0 -> trail_keep=0"""
        dp = derive(Params(perception=0.0))
        fx, _ = compute_postfx(**_default_args(dp=dp))
        assert fx.trail_keep == 0.0

    def test_trail_on_at_high_perception(self):
        dp = derive(Params(perception=1.0))
        fx, _ = compute_postfx(**_default_args(dp=dp))
        assert fx.trail_keep > 0.0

    def test_no_flash_at_low_time_arc(self):
        drift = DriftValues(time_arc=0.3, beat_phase=0.01)
        fx, _ = compute_postfx(**_default_args(drift=drift))
        assert fx.flash == 0.0

    def test_flash_at_high_time_arc_on_beat(self):
        drift = DriftValues(time_arc=0.8, beat_phase=0.02)
        fx, _ = compute_postfx(**_default_args(drift=drift, bass=0.8))
        assert fx.flash > 0.0

    def test_chaos_off_below_threshold(self):
        drift = DriftValues(chaos=0.1)
        dp = derive(Params(perception=0.0))
        fx, _ = compute_postfx(**_default_args(drift=drift, dp=dp))
        assert fx.chaos == 0.0

    def test_blend_off_at_low_blend(self):
        fx, _ = compute_postfx(**_default_args(params=Params(blend=0.0)))
        assert fx.blend_alpha == 0.0

    def test_blend_on_at_high_blend(self):
        fx, _ = compute_postfx(**_default_args(params=Params(blend=0.8)))
        assert fx.blend_alpha > 0.0

    def test_rotation_advances(self):
        """Global rotation angle should increase over time."""
        dp = derive(Params(movement=0.5))
        _, angle1 = compute_postfx(**_default_args(dp=dp, global_rot_angle=0.0))
        _, angle2 = compute_postfx(**_default_args(dp=dp, global_rot_angle=angle1))
        assert angle2 > angle1

    def test_color_hue_shift_with_color_param(self):
        dp = derive(Params(color=0.8))
        fx, _ = compute_postfx(**_default_args(dp=dp))
        # With high color, hue values should be nonzero
        assert fx.hue_r != 0.0 or fx.hue_g != 0.0 or fx.hue_b != 0.0
