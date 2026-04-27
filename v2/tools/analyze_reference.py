#!/usr/bin/env python3
"""Analyze a WAV audio file with librosa and write a .reference.json file
alongside it.

The JSON is consumed by the `reference_json_fixtures` integration test in
`kaidee-analysis`. It records raw librosa measurements so the test can
compare Kaidee's output against them throughout the whole track.

## Required tools
    pip install librosa soundfile

## Usage
    python analyze_reference.py tests/fixtures/128bpm_techno.wav
    python analyze_reference.py tests/fixtures/   # analyze every *.wav in dir
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Band definitions — must match kaidee-analysis SpectralAnalyzer bands.
# ---------------------------------------------------------------------------

BANDS: dict[str, tuple[float, float]] = {
    "sub_bass": (0.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mid": (250.0, 500.0),
    "mid": (500.0, 2000.0),
    "presence": (2000.0, 6000.0),
    "air": (6000.0, 24000.0),
}

# Each segment is this many seconds long. Shorter = finer resolution but less
# reliable per-segment BPM estimates (librosa needs a few beats per window).
SEGMENT_SECS = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_energies_from_spectrum(S, freqs, sr: float) -> dict[str, float]:
    """Compute normalized per-band energy from an STFT (short-time Fourier
    transform) magnitude matrix S (shape: n_bins × n_frames)."""
    import numpy as np  # imported here so the function is usable after the top-level import check

    energies: dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < min(hi, sr / 2.0))
        energies[name] = float(np.mean(S[mask])) if mask.any() else 0.0

    total = sum(energies.values())
    if total > 0:
        energies = {k: v / total for k, v in energies.items()}
    return {k: round(v, 4) for k, v in energies.items()}


def _std(values: list[float]) -> float:
    import numpy as np
    return float(np.std(values)) if len(values) > 1 else 0.0


# ---------------------------------------------------------------------------
# librosa analysis
# ---------------------------------------------------------------------------

def _analyze_librosa(wav_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze `wav_path` with librosa.

    Returns (global_measurements, temporal_measurements).

    global_measurements:
      bpm           — whole-track tempo estimate (BPM, beats per minute)
      rms_mean      — mean RMS (root mean square) over the whole file
      rms_max       — peak RMS value across all frames
      band_energies — normalized energy per band (sums to 1.0)
      dominant_band — band with highest mean energy

    temporal_measurements:
      segment_secs  — segment duration in seconds
      segments      — list of per-segment dicts, each with:
                        index, start_s, end_s,
                        bpm, rms_mean,
                        band_energies, dominant_band
      bpm_std       — standard deviation (SD) of BPM across segments
                      (low = stable tempo throughout the track)
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        print("ERROR: librosa not installed. Run: pip install librosa soundfile", file=sys.stderr)
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(wav_path, sr=None, mono=True)

    n_fft = 2048
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rms_frames = librosa.feature.rms(y=y, hop_length=hop)[0]

    # --- Global ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        global_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    global_bpm = round(float(np.atleast_1d(global_tempo)[0]), 2)
    global_bands = _band_energies_from_spectrum(S, freqs, sr)
    global_dominant = max(global_bands, key=global_bands.__getitem__)

    global_result: dict[str, Any] = {
        "bpm": global_bpm,
        "rms_mean": round(float(np.mean(rms_frames)), 4),
        "rms_max": round(float(np.max(rms_frames)), 4),
        "band_energies": global_bands,
        "dominant_band": global_dominant,
    }

    # --- Per-segment ---
    seg_samples = int(SEGMENT_SECS * sr)
    seg_stft_frames = int(SEGMENT_SECS * sr / hop)
    n_segments = len(y) // seg_samples

    segments: list[dict[str, Any]] = []
    bpm_per_seg: list[float] = []

    for i in range(n_segments):
        s_start = i * seg_samples
        s_end = s_start + seg_samples
        seg_y = y[s_start:s_end]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seg_tempo, _ = librosa.beat.beat_track(y=seg_y, sr=sr)
        seg_bpm = round(float(np.atleast_1d(seg_tempo)[0]), 2)
        bpm_per_seg.append(seg_bpm)

        f_start = i * seg_stft_frames
        f_end = f_start + seg_stft_frames
        seg_S = S[:, f_start:f_end]
        seg_rms = rms_frames[f_start:f_end]
        seg_bands = _band_energies_from_spectrum(seg_S, freqs, sr)
        seg_dominant = max(seg_bands, key=seg_bands.__getitem__)

        segments.append({
            "index": i,
            "start_s": round(i * SEGMENT_SECS, 1),
            "end_s": round((i + 1) * SEGMENT_SECS, 1),
            "bpm": seg_bpm,
            "rms_mean": round(float(np.mean(seg_rms)), 4),
            "band_energies": seg_bands,
            "dominant_band": seg_dominant,
        })

    temporal_result: dict[str, Any] = {
        "segment_secs": SEGMENT_SECS,
        "n_segments": n_segments,
        "segments": segments,
        "bpm_std": round(_std(bpm_per_seg), 3),
    }

    return global_result, temporal_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_file(wav_path: str) -> None:
    wav = Path(wav_path).resolve()
    if not wav.exists():
        print(f"ERROR: file not found: {wav}", file=sys.stderr)
        sys.exit(1)

    json_path = wav.with_suffix("").with_suffix(".reference.json")
    print(f"Analyzing {wav.name}…")

    print("  [librosa] global + temporal …", end=" ", flush=True)
    lib_global, lib_temporal = _analyze_librosa(str(wav))
    print(
        f"BPM={lib_global['bpm']:.1f}  "
        f"dominant={lib_global['dominant_band']}  "
        f"{lib_temporal['n_segments']} × {int(SEGMENT_SECS)} s segments  "
        f"BPM SD={lib_temporal['bpm_std']:.2f}"
    )

    output = {
        "file": wav.name,
        "analyzed_at": str(date.today()),
        "tools_used": ["librosa"],
        "per_tool": {"librosa": lib_global},
        "temporal": lib_temporal,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"  Written → {json_path.name}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze WAV files and write .reference.json reference files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="WAV_OR_DIR",
        help="WAV file(s) or directory containing WAV files",
    )
    args = parser.parse_args()

    targets: list[str] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            targets.extend(
                str(f)
                for f in sorted(path.iterdir())
                if f.suffix.lower() in {".wav", ".mp3"}
            )
        elif path.is_file():
            targets.append(str(path))
        else:
            print(f"ERROR: not a file or directory: {p}", file=sys.stderr)
            sys.exit(1)

    if not targets:
        print("No WAV files found.", file=sys.stderr)
        sys.exit(1)

    for t in targets:
        analyze_file(t)

    print(f"Done. {len(targets)} file(s) analyzed.")


if __name__ == "__main__":
    main()
