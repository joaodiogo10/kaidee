"""Drift system — shapes visuals over the duration of a set."""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriftState:
    """Mutable rolling averages for drift computation."""
    ema_energy_slow: float = 0.3
    ema_energy_fast: float = 0.3
    ema_bass_slow: float = 0.3
    onset_timestamps: list[float] = field(default_factory=list)
    # Structure detection
    energy_min_recent: float = 0.3   # rolling min over ~4 bars
    drop_decay: float = 0.0          # decaying impulse after a drop


@dataclass
class DriftValues:
    """Output of the drift system for a single frame."""
    intensity: float = 1.0
    warmth: float = 0.0
    code_mix: float = 0.7
    chaos: float = 0.0
    vignette: float = 0.75
    mode_energy: float = 0.5
    time_arc: float = 0.5
    music_intensity: float = 0.5
    energy_delta: float = 0.0   # +buildup, -breakdown
    drop_impact: float = 0.0    # 1.0 on drop, decays to 0
    bpm: float = 128.0
    bar_idx: int = 0
    bar_phase: float = 0.0
    beat_phase: float = 0.0
    beat_pulse: float = 0.0   # 1.0 on beat, decays to 0
    bar_pulse: float = 0.0    # 1.0 on downbeat, decays to 0


DRIFT_DISABLED = DriftValues()


def compute_drift(
    t: float,
    fps: float,
    set_duration: float,
    bass: float,
    energy: float,
    onset: float,
    audio_source,
    state: DriftState,
) -> DriftValues:
    """Compute drift values from time, audio, and rolling state."""
    bpm = audio_source.get_bpm(t)
    _, beat_phase = audio_source.get_beat_phase(t)
    bar_idx, bar_phase = audio_source.get_bar(t)
    bpm_factor = bpm / 128.0

    dt = 1.0 / max(fps, 1)

    # Rolling intensity averages
    alpha_slow = min(1.0, dt / 30.0)
    state.ema_energy_slow += alpha_slow * (energy - state.ema_energy_slow)
    state.ema_bass_slow += alpha_slow * (bass - state.ema_bass_slow)

    alpha_fast = min(1.0, dt / 3.0)  # ~3s window for faster structure response
    state.ema_energy_fast += alpha_fast * (energy - state.ema_energy_fast)

    # Onset rate
    if onset > 0.5:
        state.onset_timestamps.append(t)
    state.onset_timestamps = [ts for ts in state.onset_timestamps if t - ts < 10]
    onset_rate = len(state.onset_timestamps) / 10.0

    # Time arc
    frac = min(1.0, t / max(set_duration, 1.0))
    if frac < 0.10:
        time_arc = 0.3 + frac / 0.10 * 0.2
    elif frac < 0.40:
        time_arc = 0.5 + (frac - 0.10) / 0.30 * 0.3
    elif frac < 0.70:
        time_arc = 0.8 + (frac - 0.40) / 0.30 * 0.2
    elif frac < 0.85:
        time_arc = 1.0 - (frac - 0.70) / 0.15 * 0.3
    else:
        time_arc = 0.7 + (frac - 0.85) / 0.15 * 0.3

    # Music intensity
    music_intensity = state.ema_energy_slow * 0.6 + state.ema_bass_slow * 0.4
    intensity_delta = state.ema_energy_fast - state.ema_energy_slow

    # Structure detection: breakdown/buildup/drop
    energy_delta = np.clip(intensity_delta * 3.0, -1.0, 1.0)

    # Track recent energy minimum (slow decay toward current energy)
    alpha_min = min(1.0, dt / 8.0)  # ~8s window
    if energy < state.energy_min_recent:
        state.energy_min_recent = energy  # snap to new lows
    else:
        state.energy_min_recent += alpha_min * (energy - state.energy_min_recent)

    # Drop detection: energy jumps well above recent minimum
    energy_jump = state.ema_energy_fast - state.energy_min_recent
    if energy_jump > 0.15 and intensity_delta > 0.05 and bass > 0.35:
        state.drop_decay = min(1.0, state.drop_decay + 0.6)
        state.energy_min_recent = state.ema_energy_fast  # reset baseline
    else:
        state.drop_decay *= max(0, 1.0 - dt * 1.5)  # decay over ~0.7s
    drop_impact = state.drop_decay

    # Intensity — boosted by drops and buildups
    intensity_raw = (time_arc * (0.7 + music_intensity * 0.6)
                     + intensity_delta * 0.3
                     + drop_impact * 0.5)
    intensity = 0.5 + np.clip(intensity_raw, 0, 1) * 1.3

    # Warmth
    warmth_period = 120.0 / (bpm_factor * (0.5 + music_intensity))
    warmth = np.sin(2 * np.pi * t / warmth_period) * (0.15 + time_arc * 0.2 + music_intensity * 0.15)

    # Chaos — spikes on drops and buildups
    chaos = np.clip(time_arc * 0.4 + music_intensity * 0.4
                    + max(0, intensity_delta) * 0.2
                    + drop_impact * 0.4, 0, 1)

    # Beat pulses (decay curves synced to BPM)
    beat_pulse = max(0, 1.0 - beat_phase * 4) ** 2
    bar_pulse = max(0, 1.0 - bar_phase * 2) ** 2

    # Code mix
    code_base = 0.3 + time_arc * 0.2 + music_intensity * 0.3
    code_mix = np.clip(code_base + beat_pulse * 0.2 * time_arc, 0.2, 1.0)

    # Vignette — tightens during breakdowns, opens on drops
    breakdown_factor = max(0, -energy_delta) * 0.2
    vignette = np.clip(1.0 - time_arc * 0.3 - music_intensity * 0.2
                        + breakdown_factor - drop_impact * 0.15, 0.3, 0.9)

    # Mode energy
    mode_energy = np.clip(music_intensity * 0.5 + time_arc * 0.3 + onset_rate * 0.1, 0, 1)

    return DriftValues(
        intensity=float(intensity),
        warmth=float(warmth),
        code_mix=float(code_mix),
        chaos=float(chaos),
        vignette=float(vignette),
        mode_energy=float(mode_energy),
        time_arc=float(time_arc),
        music_intensity=float(music_intensity),
        energy_delta=float(energy_delta),
        drop_impact=float(drop_impact),
        bpm=float(bpm),
        bar_idx=int(bar_idx),
        bar_phase=float(bar_phase),
        beat_phase=float(beat_phase),
        beat_pulse=float(beat_pulse),
        bar_pulse=float(bar_pulse),
    )
