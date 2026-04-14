"""Audio sources: live mic, file playback, and synthetic demo."""

from __future__ import annotations
from typing import Protocol
import logging
import sys
import tempfile
import threading
import wave

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import pygame
except ImportError:
    pygame = None

logger = logging.getLogger(__name__)


def _get_bar(get_beat_phase, t: float, beats_per_bar: int = 4) -> tuple[int, float]:
    """Compute bar index and phase from beat phase. Shared by all audio sources."""
    beat_idx, beat_phase = get_beat_phase(t)
    bar_idx = beat_idx // beats_per_bar
    bar_beat = beat_idx % beats_per_bar
    bar_phase = (bar_beat + beat_phase) / beats_per_bar
    return bar_idx, bar_phase


class AudioSource(Protocol):
    """Common interface for all audio sources."""
    bass: float
    mids: float
    highs: float
    energy: float
    onset: float

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def update(self, t: float) -> None: ...
    def get_bpm(self, t: float | None = None) -> float: ...
    def get_beat_phase(self, t: float) -> tuple[int, float]: ...
    def get_bar(self, t: float, beats_per_bar: int = 4) -> tuple[int, float]: ...


class AudioAnalyzer:
    """Realtime audio analysis via sounddevice callback."""

    def __init__(self, device=None, samplerate: int = 44100, blocksize: int = 2048):
        self.sr = samplerate
        self.blocksize = blocksize
        self.bass = 0.0
        self.mids = 0.0
        self.highs = 0.0
        self.energy = 0.0
        self.onset = 0.0
        self._prev_energy = 0.0
        self._peak_bass = 1e-6
        self._peak_mids = 1e-6
        self._peak_highs = 1e-6
        self._stream = None
        self._bpm = 128.0
        self._bpm_display = 128
        self._last_t = 0.0
        # Raw audio ring buffer for BPM (~8 seconds)
        self._bpm_buf_len = samplerate * 8
        self._bpm_buf = np.zeros(self._bpm_buf_len, dtype=np.float32)
        self._bpm_buf_pos = 0
        self._bpm_filled = 0
        self._bpm_counter = 0
        self._bpm_lock = threading.Lock()
        self._bpm_running = False
        self._bpm_votes: list[int] = []
        # Recording buffer
        self._recording = False
        self._rec_chunks: list[np.ndarray] = []

        if sd is not None:
            try:
                self._stream = sd.InputStream(
                    device=device, channels=1, samplerate=samplerate,
                    blocksize=blocksize, callback=self._callback,
                )
            except Exception as e:
                print(f"WARNING: Could not open audio device: {e}")
                print("  Falling back to demo audio.")

    def _callback(self, indata, frames, time_info, status):
        audio = indata[:, 0]

        # Buffer audio for recording
        if self._recording:
            self._rec_chunks.append(audio.copy())

        window = np.hanning(len(audio))
        fft = np.abs(np.fft.rfft(audio * window))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sr)

        bass_e = np.mean(fft[freqs < 300])
        mid_e = np.mean(fft[(freqs >= 300) & (freqs < 4000)])
        high_e = np.mean(fft[freqs >= 4000])

        self._peak_bass = max(self._peak_bass * 0.998, bass_e, 1e-6)
        self._peak_mids = max(self._peak_mids * 0.998, mid_e, 1e-6)
        self._peak_highs = max(self._peak_highs * 0.998, high_e, 1e-6)
        self.bass = min(1.0, bass_e / self._peak_bass)
        self.mids = min(1.0, mid_e / self._peak_mids)
        self.highs = min(1.0, high_e / self._peak_highs)

        new_e = np.sqrt(np.mean(audio ** 2))
        self.onset = min(1.0, max(0, new_e - self._prev_energy) * 10)
        self._prev_energy = new_e * 0.8 + self._prev_energy * 0.2
        self.energy = min(1.0, new_e * 5)

        # Fill ring buffer with raw audio for BPM
        chunk_len = len(audio)
        end = self._bpm_buf_pos + chunk_len
        if end <= self._bpm_buf_len:
            self._bpm_buf[self._bpm_buf_pos:end] = audio
        else:
            first = self._bpm_buf_len - self._bpm_buf_pos
            self._bpm_buf[self._bpm_buf_pos:] = audio[:first]
            self._bpm_buf[:chunk_len - first] = audio[first:]
        self._bpm_buf_pos = end % self._bpm_buf_len
        self._bpm_filled = min(self._bpm_filled + chunk_len, self._bpm_buf_len)

        # Run BPM detection every ~3 seconds in background thread
        self._bpm_counter += chunk_len
        if self._bpm_counter >= self.sr * 1 and self._bpm_filled >= self.sr * 3 and not self._bpm_running:
            self._bpm_counter = 0
            # Snapshot the buffer for the background thread
            if self._bpm_filled < self._bpm_buf_len:
                snap = self._bpm_buf[:self._bpm_filled].copy()
            else:
                snap = np.roll(self._bpm_buf, -self._bpm_buf_pos).copy()
            self._bpm_running = True
            threading.Thread(target=self._update_bpm, args=(snap,), daemon=True).start()

    def _update_bpm(self, raw: np.ndarray) -> None:
        """Estimate BPM via bass energy autocorrelation (very fast, no STFT)."""
        try:
            # Decimate with averaging (proper anti-alias) → ~2756 Hz
            # Keeps bass + low-mids (<1378 Hz) — good for kick + snare
            dec = 16
            n_full = (len(raw) // dec) * dec
            ds = raw[:n_full].reshape(-1, dec).mean(axis=1)
            ds_sr = float(self.sr) / dec  # ~2756 Hz

            # Squared energy in small blocks → ~345 Hz envelope
            # Higher rate = better precision at high BPMs
            block = max(1, int(ds_sr / 345))
            n_blocks = len(ds) // block
            if n_blocks < 50:
                return
            energy = (ds[:n_blocks * block].reshape(n_blocks, block) ** 2).mean(axis=1)
            env_rate = ds_sr / block  # ~345 Hz

            # Onset: positive energy increase (kick transients)
            onset = np.maximum(0, np.diff(energy))

            # --- FFT-based autocorrelation ---
            onset = onset - onset.mean()
            n = len(onset)
            fft_size = 1 << int(np.ceil(np.log2(2 * n)))
            F = np.fft.rfft(onset, n=fft_size)
            corr = np.fft.irfft(F * np.conj(F))[:n]
            if corr[0] <= 0:
                return
            corr /= corr[0]

            # Search in 20-200 BPM range
            min_lag = max(1, int(env_rate * 60.0 / 200))
            max_lag = min(n - 2, int(env_rate * 60.0 / 20))
            if max_lag <= min_lag:
                return

            search = corr[min_lag:max_lag + 1]
            peak_idx = int(np.argmax(search))
            peak_val = float(search[peak_idx])
            peak_lag = min_lag + peak_idx

            # Octave check: if half-lag (double tempo) has a decent peak, prefer it
            half_lag = peak_lag // 2
            if half_lag >= min_lag:
                half_idx = half_lag - min_lag
                if 0 < half_idx < len(search) - 1:
                    half_val = float(search[half_idx])
                    if half_val > peak_val * 0.5:
                        peak_idx = half_idx
                        peak_lag = half_lag

            # Parabolic interpolation for sub-sample precision
            peak_lag = float(peak_lag)
            if 0 < peak_idx < len(search) - 1:
                y0, y1, y2 = float(search[peak_idx - 1]), float(search[peak_idx]), float(search[peak_idx + 1])
                denom = 2.0 * (2.0 * y1 - y0 - y2)
                if abs(denom) > 1e-10:
                    peak_lag = (peak_lag - peak_idx) + peak_idx + (y0 - y2) / denom

            raw_bpm = 60.0 * env_rate / peak_lag

            # Normalize to 20-200 range
            while raw_bpm < 20:
                raw_bpm *= 2
            while raw_bpm > 200:
                raw_bpm /= 2

            vote = round(raw_bpm)

            with self._bpm_lock:
                self._bpm_votes.append(vote)
                if len(self._bpm_votes) > 8:
                    self._bpm_votes = self._bpm_votes[-8:]

                if len(self._bpm_votes) >= 2:
                    tol = max(3, self._bpm_display * 0.03)

                    # Check last 3 votes for consensus (fast switch)
                    recent = self._bpm_votes[-3:] if len(self._bpm_votes) >= 3 else self._bpm_votes[-2:]
                    if max(recent) - min(recent) <= tol:
                        median_bpm = round(float(np.median(recent)))
                        if self._bpm_display == 128 or abs(median_bpm - self._bpm_display) > tol:
                            self._bpm_display = median_bpm
                            self._bpm = float(median_bpm)
        except Exception:
            logger.debug("BPM detection error", exc_info=True)
        finally:
            self._bpm_running = False

    @property
    def duration(self) -> float:
        return 300.0

    def start(self) -> None:
        if self._stream is not None:
            self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()

    def start_recording(self) -> None:
        """Start buffering raw audio for recording."""
        self._rec_chunks.clear()
        self._recording = True

    def stop_recording(self) -> str | None:
        """Stop buffering and save to a temporary WAV file. Returns the path."""
        self._recording = False
        if not self._rec_chunks:
            return None
        audio = np.concatenate(self._rec_chunks)
        self._rec_chunks.clear()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            pcm = (audio * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
        return tmp.name

    def update(self, t: float) -> None:
        self._last_t = t

    def get_bpm(self, t: float | None = None) -> float:
        return float(self._bpm_display)

    def get_beat_phase(self, t: float) -> tuple[int, float]:
        beat_dur = 60.0 / self._bpm
        beat_idx = int(t / beat_dur)
        phase = (t % beat_dur) / beat_dur
        return beat_idx, phase

    def get_bar(self, t: float, beats_per_bar: int = 4) -> tuple[int, float]:
        return _get_bar(self.get_beat_phase, t, beats_per_bar)


class DemoAudio:
    """Synthetic 128 BPM audio for testing without hardware."""

    def __init__(self):
        self.bass = 0.0
        self.mids = 0.0
        self.highs = 0.0
        self.energy = 0.0
        self.onset = 0.0
        self._bpm = 128
        self._beat_dur = 60.0 / self._bpm

    @property
    def duration(self) -> float:
        return 300.0

    def update(self, t: float) -> None:
        beat_phase = (t % self._beat_dur) / self._beat_dur
        kick = max(0, 1.0 - beat_phase * 6) ** 2
        hihat = max(0, 1.0 - ((t + self._beat_dur / 2) % self._beat_dur) / self._beat_dur * 8) ** 2
        self.bass = kick * 0.9
        self.mids = 0.3 + 0.3 * np.sin(t * 0.7)
        self.highs = hihat * 0.7
        self.energy = 0.3 + kick * 0.5
        self.onset = 1.0 if beat_phase < 0.03 else 0.0

    def get_bpm(self, t: float | None = None) -> float:
        return float(self._bpm)

    def get_beat_phase(self, t: float) -> tuple[int, float]:
        beat_idx = int(t / self._beat_dur)
        phase = (t % self._beat_dur) / self._beat_dur
        return beat_idx, phase

    def get_bar(self, t: float, beats_per_bar: int = 4) -> tuple[int, float]:
        return _get_bar(self.get_beat_phase, t, beats_per_bar)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class FileAudio:
    """Pre-analyzed audio file with librosa, playback via pygame.mixer."""

    def __init__(self, filepath: str):
        if librosa is None:
            print("ERROR: librosa required for --play. Install: pip install librosa")
            sys.exit(1)

        self.filepath = filepath
        self.bass = 0.0
        self.mids = 0.0
        self.highs = 0.0
        self.energy = 0.0
        self.onset = 0.0
        self.paused = False
        self._duration = 0.0
        self._seek_offset = 0.0
        self._volume = 1.0
        self._pause_pos = 0.0

        print(f"Analyzing audio: {filepath}...")
        y, sr = librosa.load(filepath, sr=22050, mono=True)
        self._duration = len(y) / sr

        hop = 512
        self._hop = hop
        self._sr = sr
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-10)

        n_frames = S_norm.shape[1]
        self._times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)

        def _band_norm(arr):
            band = np.mean(arr, axis=0)
            lo, hi = band.min(), band.max()
            return (band - lo) / (hi - lo + 1e-10)

        self._bass = _band_norm(S_norm[:16])
        self._mids = _band_norm(S_norm[16:80])
        self._highs = _band_norm(S_norm[80:])

        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        self._energy = rms / (rms.max() + 1e-10)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        self._onset = onset_env / (onset_env.max() + 1e-10)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, start_bpm=130)
        self._bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        self._bpm = np.clip(self._bpm, 60, 200)
        self._beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
        print(f"  BPM: {self._bpm:.1f}, {len(self._beat_times)} beats detected")

        # Ensure all arrays are same length
        min_len = min(len(self._bass), len(self._mids), len(self._highs),
                      len(self._energy), len(self._onset), len(self._times))
        self._bass = self._bass[:min_len]
        self._mids = self._mids[:min_len]
        self._highs = self._highs[:min_len]
        self._energy = self._energy[:min_len]
        self._onset = self._onset[:min_len]
        self._times = self._times[:min_len]
        self._smoothed_bpm = self._bpm

        print(f"  Duration: {self._duration:.1f}s, {n_frames} analysis frames")

    @property
    def duration(self) -> float:
        return self._duration

    def _frame_at(self, t: float) -> int:
        idx = np.searchsorted(self._times, t)
        return min(idx, len(self._times) - 1)

    def update(self, t: float) -> None:
        idx = self._frame_at(t)
        self.bass = float(self._bass[idx])
        self.mids = float(self._mids[idx])
        self.highs = float(self._highs[idx])
        self.energy = float(self._energy[idx])
        self.onset = float(self._onset[idx])

    def get_time(self) -> float:
        if not pygame.mixer.get_init():
            return 0.0
        if self.paused:
            return self._pause_pos
        pos_ms = pygame.mixer.music.get_pos()
        if pos_ms < 0:
            return self._seek_offset
        return self._seek_offset + pos_ms / 1000.0

    def start(self) -> None:
        pygame.mixer.quit()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.music.load(self.filepath)
        pygame.mixer.music.play()
        self._seek_offset = 0.0
        self._volume = 1.0

    def stop(self) -> None:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()

    def toggle_pause(self) -> None:
        if self.paused:
            pygame.mixer.music.unpause()
        else:
            pygame.mixer.music.pause()
            self._pause_pos = self.get_time()
        self.paused = not self.paused

    def seek(self, target_s: float) -> None:
        target_s = max(0.0, min(target_s, self._duration - 0.1))
        if not pygame.mixer.get_init():
            return
        was_paused = self.paused
        pygame.mixer.music.stop()
        pygame.mixer.music.load(self.filepath)
        pygame.mixer.music.play(start=target_s)
        pygame.mixer.music.set_volume(self._volume)
        self._seek_offset = target_s
        if was_paused:
            pygame.mixer.music.pause()
            self.paused = True
            self._pause_pos = target_s

    def seek_relative(self, delta_s: float) -> None:
        current = self.get_time()
        self.seek(current + delta_s)

    def set_volume(self, vol: float) -> None:
        self._volume = max(0.0, min(1.0, vol))
        if pygame.mixer.get_init():
            pygame.mixer.music.set_volume(self._volume)

    def get_volume(self) -> float:
        return self._volume

    def get_bpm(self, t: float | None = None) -> float:
        if t is None or len(self._beat_times) < 4:
            return self._bpm
        idx = np.searchsorted(self._beat_times, t)
        lo = max(0, idx - 8)
        hi = min(len(self._beat_times), idx + 8)
        if hi - lo < 3:
            return self._bpm
        intervals = np.diff(self._beat_times[lo:hi])
        sorted_iv = np.sort(intervals)
        trim = max(1, len(sorted_iv) // 4)
        if len(sorted_iv) > 2 * trim:
            avg_interval = np.mean(sorted_iv[trim:-trim])
        else:
            avg_interval = np.mean(sorted_iv)
        if avg_interval > 0:
            raw = float(np.clip(60.0 / avg_interval, 60, 200))
            self._smoothed_bpm += 0.05 * (raw - self._smoothed_bpm)
            return round(self._smoothed_bpm, 1)
        return self._bpm

    def get_beat_phase(self, t: float) -> tuple[int, float]:
        if len(self._beat_times) < 2:
            beat_dur = 60.0 / self._bpm
            beat_idx = int(t / beat_dur)
            phase = (t % beat_dur) / beat_dur
            return beat_idx, phase
        idx = np.searchsorted(self._beat_times, t) - 1
        idx = max(0, min(idx, len(self._beat_times) - 2))
        beat_start = self._beat_times[idx]
        beat_end = self._beat_times[idx + 1]
        dur = beat_end - beat_start
        phase = (t - beat_start) / dur if dur > 0 else 0.0
        return idx, max(0.0, min(1.0, phase))

    def get_bar(self, t: float, beats_per_bar: int = 4) -> tuple[int, float]:
        return _get_bar(self.get_beat_phase, t, beats_per_bar)


def list_devices() -> None:
    if sd is not None:
        print(sd.query_devices())
    else:
        print("sounddevice not installed")


# Known virtual audio loopback device name patterns
_LOOPBACK_PATTERNS = [
    "blackhole",      # macOS (free, open source)
    "loopback",       # macOS (Rogue Amoeba, paid)
    "soundflower",    # macOS (legacy)
    "monitor of",     # Linux PulseAudio/PipeWire
    "cable output",   # Windows VB-Cable
    "stereo mix",     # Windows built-in (some drivers)
    "what u hear",    # Windows (some Realtek drivers)
]


def find_loopback_device() -> int | None:
    """Auto-detect a virtual audio loopback device for system audio capture.

    Returns the device index if found, None otherwise.
    """
    if sd is None:
        return None

    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        # Only consider input-capable devices
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"].lower()
        for pattern in _LOOPBACK_PATTERNS:
            if pattern in name:
                print(f"Loopback device found: [{idx}] {dev['name']}")
                return idx
    return None
