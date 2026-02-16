"""Shared post-processing parameter computation for GPU and CPU paths."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from src.params import Params, DerivedParams
from src.drift import DriftValues


@dataclass
class PostFX:
    """Computed post-processing effect parameters."""
    # Pan (normalized fractions; GPU uses directly, CPU scales by pixel dims)
    pan_x: float = 0.0
    pan_y: float = 0.0
    # Zoom
    zoom: float = 1.0
    # Rotation state (degrees, radians)
    rot_angle_deg: float = 0.0
    rot_wobble_rad: float = 0.0
    rot_shift_x: float = 0.0
    rot_shift_y: float = 0.0
    # Color
    saturation: float = 1.0
    contrast: float = 0.0
    hue_r: float = 0.0
    hue_g: float = 0.0
    hue_b: float = 0.0
    # Effects
    flash: float = 0.0
    chaos: float = 0.0
    chaos_spacing: int = 0
    chroma_h: float = 0.0
    chroma_v: float = 0.0
    brightness: float = 1.0
    trail_keep: float = 0.0
    vig_strength: float = 0.75
    # Blend
    blend_alpha: float = 0.0


# Drop reaction flavors — each drop picks one based on bar_idx.
# Weights: (saturation, contrast, flash, chroma, zoom, brightness)
_DROP_FLAVORS = [
    (0.5, 0.3, 0.6, 1.2, 0.08, 0.2),   # full impact
    (0.0, 0.0, 0.8, 0.3, 0.0, 0.3),     # flash + bright, no color
    (0.7, 0.4, 0.0, 1.5, 0.12, 0.0),    # color burst, no flash
    (0.0, 0.0, 0.0, 0.0, 0.15, 0.0),    # zoom only
    (0.3, 0.5, 0.4, 0.5, 0.0, -0.15),   # contrast + darken
    (0.8, 0.0, 0.0, 2.0, 0.0, 0.0),     # chroma + saturation
]


def compute_postfx(
    t: float,
    fps: float,
    dp: DerivedParams,
    drift: DriftValues,
    bass: float,
    energy: float,
    params: Params,
    global_rot_angle: float,
) -> tuple[PostFX, float]:
    """Compute post-processing parameters.

    Returns (PostFX, updated_global_rot_angle).

    Pan/rotation values are in GPU-normalized space.  The CPU path
    applies its own scale multipliers for pixel-space conversion.
    """
    ds = dp.drift_speed
    buildup = max(0, drift.energy_delta)
    drop = drift.drop_impact
    speed_mult = 0.5 + ds * 6.0
    fx = PostFX()

    # Pick a drop flavor based on bar index (varies per section)
    flavor = _DROP_FLAVORS[drift.bar_idx % len(_DROP_FLAVORS)]
    d_sat, d_con, d_flash, d_chroma, d_zoom, d_bright = flavor

    # ---- Pan (GPU-normalized fractions) ----
    pan_pct = 0.002 + ds * 0.05
    fx.pan_x = (np.sin(t * 0.23 * speed_mult) * pan_pct
                + np.sin(t * 0.11 * speed_mult) * pan_pct * 0.5
                + (params.pan_x - 0.5) * 0.5)
    fx.pan_y = (np.cos(t * 0.17 * speed_mult) * pan_pct * 0.7
                + np.cos(t * 0.09 * speed_mult) * pan_pct * 0.3
                + (params.pan_y - 0.5) * 0.5)

    # ---- Zoom ----
    breath_amp = 0.002 + ds * 0.01
    fx.zoom = 1.0 + breath_amp * max(0.0, np.sin(t * (0.4 + ds * 1.5)))
    fx.zoom += drop * d_zoom
    fx.zoom *= 0.5 + params.zoom * 1.5  # manual zoom

    # ---- Rotation ----
    rot_speed_dps = ds * 50
    new_rot_angle = global_rot_angle + rot_speed_dps / max(fps, 1)
    fx.rot_angle_deg = new_rot_angle
    fx.rot_wobble_rad = float(np.radians(
        new_rot_angle + np.sin(t * 0.5 * speed_mult) * ds * 10))
    if ds > 0.05:
        fx.rot_shift_x = float(np.sin(fx.rot_wobble_rad) * 0.03 * ds)
        fx.rot_shift_y = float(np.cos(fx.rot_wobble_rad) * 0.03 * ds)

    # ---- Color ----
    color_int = dp.color_intensity
    intensity = drift.intensity * color_int
    warmth = drift.warmth
    cs = dp.color_shift_speed
    sat_mod = 1.0 + drop * d_sat
    fx.saturation = intensity * sat_mod
    fx.contrast = max(0.0, (intensity - 1.2) * 0.8) if intensity > 1.2 else 0.0
    fx.contrast += drop * d_con
    shift = warmth * 80 * (0.5 + cs * 0.5) / 255.0
    fx.hue_r = float(np.sin(t * cs * 0.5) * cs * 40 / 255.0 + shift)
    fx.hue_g = float(np.sin(t * cs * 0.7 + 2.1) * cs * 30 / 255.0 + shift * 0.3)
    fx.hue_b = float(np.sin(t * cs * 0.3 + 4.2) * cs * 45 / 255.0 - shift)

    # ---- Chaos + flash ----
    chaos_raw = dp.chaos_manual if dp.chaos_manual >= 0 else drift.chaos
    if drift.time_arc > 0.6 and drift.beat_phase < 0.08:
        bass_g = min(1.0, bass * dp.bass_gain)
        fx.flash = (drift.time_arc - 0.6) * 0.4 * bass_g * 80 / 255.0
    if drop > 0.2 and d_flash > 0:
        fx.flash = max(fx.flash, drop * d_flash * 80 / 255.0)
    fx.chaos = max(0.0, (chaos_raw - 0.2) * 50 / 255.0) if chaos_raw > 0.2 else 0.0
    fx.chaos_spacing = max(2, 6 - int(chaos_raw * 4)) if chaos_raw > 0.5 else 0

    # ---- Chromatic aberration (raw pixel amounts; GPU divides by res) ----
    if dp.chroma > 0.01:
        bass_g = min(1.0, bass * dp.bass_gain)
        energy_g = min(1.0, energy * (dp.bass_gain + dp.mids_gain) / 2)
        chroma_boost = 1.0 + buildup * 0.8 + drop * d_chroma
        chroma_amount = (dp.chroma * 0.4
                         + dp.chroma * (bass_g * 0.8 + energy_g * 0.3)
                         * (1.0 + drift.chaos * 0.5)) * chroma_boost
        if chroma_amount > 0.05:
            fx.chroma_h = chroma_amount * 20
            fx.chroma_v = chroma_amount * 8

    # ---- Brightness ----
    br = params.brightness
    fx.brightness = float(3.0 ** (br * 2 - 1)) if abs(br - 0.5) > 0.02 else 1.0
    fx.brightness *= (1.0 + drop * d_bright)

    # ---- Trail ----
    fx.trail_keep = dp.trail * 0.7 if dp.trail > 0.01 else 0.0

    # ---- Vignette ----
    fx.vig_strength = drift.vignette

    # ---- Blend ----
    if params.blend > 0.05:
        fx.blend_alpha = float(min(
            params.blend * (0.3 + energy * 0.5 + bass * 0.3), 0.95))

    return fx, new_rot_angle
