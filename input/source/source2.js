"""Tests for src.drift."""

import numpy as np

from src.drift import compute_drift, DriftState, DriftValues, DRIFT_DISABLED


class MockAudio:
    """Minimal audio source for drift tests."""
    def __init__(self, bpm=128.0):
        self._bpm = bpm

    def get_bpm(self, t):
        return self._bpm

    def get_beat_phase(self, t):
        beat_dur = 60.0 / self._bpm
        beat_idx = int(t / beat_dur)
        phase = (t % beat_dur) / beat_dur
        return beat_idx, phase

    def get_bar(self, t, beats_per_bar=4):
        beat_idx, phase = self.get_beat_phase(t)
        return beat_idx // 4, (beat_idx % 4 + phase) / 4


class TestTimeArc:
    def _drift_at_frac(self, frac, duration=100.0):
        t = frac * duration
        state = DriftState()
        audio = MockAudio()
        return compute_drift(t, 60.0, duration, 0.3, 0.3, 0.0, audio, state)

    def test_intro(self):
        dv = self._drift_at_frac(0.05)
        assert 0.3 <= dv.time_arc <= 0.55

    def test_ramp(self):
        dv = self._drift_at_frac(0.25)
        assert 0.5 <= dv.time_arc <= 0.85

    def test_peak(self):
        dv = self._drift_at_frac(0.55)
        assert 0.85 <= dv.time_arc <= 1.0

    def test_dip(self):
        dv = self._drift_at_frac(0.75)
        assert 0.7 <= dv.time_arc <= 0.95

    def test_finale(self):
        dv = self._drift_at_frac(0.95)
        assert 0.85 <= dv.time_arc <= 1.05


class TestBeatPulse:
    def test_on_beat(self):
        """At t=0 (exactly on a beat), beat_phase=0 -> pulse=1.0."""
        state = DriftState()
        audio = MockAudio()
        dv = compute_drift(0.0, 60.0, 100.0, 0.3, 0.3, 0.0, audio, state)
        assert dv.beat_pulse == 1.0

    def test_off_beat(self):
        """At half a beat, pulse should be near zero."""
        state = DriftState()
        audio = MockAudio()
        # half-beat: t = 60/128/2 = 0.234375
        half_beat = 60.0 / 128.0 / 2.0
        dv = compute_drift(half_beat, 60.0, 100.0, 0.3, 0.3, 0.0, audio, state)
        assert dv.beat_pulse < 0.05


class TestChaos:
    def test_low_energy_low_chaos(self):
        state = DriftState()
        audio = MockAudio()
        dv = compute_drift(5.0, 60.0, 100.0, 0.1, 0.1, 0.0, audio, state)
        assert dv.chaos < 0.4

    def test_high_energy_higher_chaos(self):
        state = DriftState()
        audio = MockAudio()
        # Pump EMA by running many frames with high energy
        for i in range(200):
            t = 50.0 + i / 60.0  # mid-set for high time_arc
            compute_drift(t, 60.0, 100.0, 0.9, 0.9, 0.8, audio, state)
        dv = compute_drift(53.5, 60.0, 100.0, 0.9, 0.9, 0.8, audio, state)
        assert dv.chaos > 0.5


class TestEMAConvergence:
    def test_converges_to_input(self):
        state = DriftState()
        audio = MockAudio()
        target_energy = 0.8
        # Slow EMA has 30s time constant; need many frames to converge
        for i in range(3000):
            t = i / 60.0
            compute_drift(t, 60.0, 100.0, target_energy, target_energy, 0.0, audio, state)
        assert abs(state.ema_energy_slow - target_energy) < 0.15
        assert abs(state.ema_energy_fast - target_energy) < 0.05


class TestOnsetWindow:
    def test_onsets_expire(self):
        state = DriftState()
        audio = MockAudio()
        # Add onsets at t=1
        compute_drift(1.0, 60.0, 100.0, 0.5, 0.5, 0.8, audio, state)
        assert len(state.onset_timestamps) > 0
        # Advance past 10s window
        compute_drift(12.0, 60.0, 100.0, 0.5, 0.5, 0.0, audio, state)
        assert len(state.onset_timestamps) == 0


class TestDriftDisabled:
    def test_defaults(self):
        d = DRIFT_DISABLED
        assert d.intensity == 1.0
        assert d.warmth == 0.0
        assert d.chaos == 0.0
        assert d.beat_pulse == 0.0
        assert d.bpm == 128.0


class TestDriftOutput:
    def test_returns_drift_values(self):
        state = DriftState()
        audio = MockAudio()
        dv = compute_drift(1.0, 60.0, 100.0, 0.5, 0.5, 0.3, audio, state)
        assert isinstance(dv, DriftValues)

    def test_bpm_matches_audio(self):
        state = DriftState()
        audio = MockAudio(bpm=140.0)
        dv = compute_drift(1.0, 60.0, 100.0, 0.5, 0.5, 0.0, audio, state)
        assert dv.bpm == 140.0
