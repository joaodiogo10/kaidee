# FftAnalyzer — Short-Time FFT with Hann Window

Converts a window of audio samples into a **magnitude spectrum** and a set of
**frequency-band energies** via a short-time FFT (fast Fourier transform).

```
sample window (2048 samples)
        │
        ▼
  Hann windowing          ← tapers frame edges to zero
        │
        ▼
   forward FFT            ← complex spectrum, FFT_SIZE bins
        │
        ▼
  magnitude + normalize   ← positive-frequency half only (1024 bins)
        │
   ┌────┴─────────────────────────────────────────────────────────┐
   ▼                                                              ▼
band energies (6 bands)                                    magnitudes slice
  → AudioFrame fields                                 → dynamics, beat tracker
```

## Trade-offs

| Property              | Value                                              |
|-----------------------|----------------------------------------------------|
| Frequency resolution  | ~21.5 Hz/bin at 44 100 Hz (coarse in bass range)  |
| Window latency        | ~46 ms (FFT_SIZE = 2048 samples at 44 100 Hz)     |
| CPU cost              | O(N log N) per frame — cheap at 100 Hz rate       |
| Spectral leakage      | Reduced by Hann window; non-integer bins still spill |

## Hann Window

Before running the FFT, each sample is multiplied by a Hann (raised-cosine)
weight:

```
w[i] = 0.5 · (1 − cos(2π·i / (N−1)))
```

This tapers the frame to zero at both edges. Without it, the sharp jump between
the last sample of one frame and the start of the next is read by the FFT as
a high-frequency discontinuity — producing artificial spectral content
("spectral leakage") that corrupts every bin.

## FFT and Magnitude Normalization

The FFT operates on `FFT_SIZE = 2048` samples. For a real-valued input signal,
the output is symmetric: the second half of the spectrum mirrors the first
(Nyquist symmetry). Only the first `N/2 = 1024` bins are used.

Each bin's magnitude is normalized by `FFT_SIZE` so the values don't grow with
window length:

```
magnitude[i] = |fft_output[i]| × 2 / FFT_SIZE
```

The `× 2` compensates for discarding the mirrored half.

Each bin covers `(sample_rate / 2) / 1024` Hz — about **21.5 Hz per bin** at
44 100 Hz.

## Frequency Bands

Six bands partition the audible spectrum. Band energy is the **sum of squared
magnitudes** within the bin range, then square-rooted (i.e. the Euclidean norm
of the magnitude vector):

```
band_energy = √( Σ magnitude[i]² )   for i in [lo_bin, hi_bin)
```

Sum-of-squares (rather than mean) is intentional: wide bands like AIR (744
bins) should register more total energy than narrow bands like SUB (2 bins)
when the spectrum is uniformly active. Mean would compress them all to the
same scale.

| Band       | Range         | Approx. bin count | Content                      |
|------------|---------------|-------------------|------------------------------|
| `sub_bass` | 20 – 60 Hz    | 2                 | Sub-bass, rumble             |
| `bass`     | 60 – 250 Hz   | 9                 | Kick, bass guitar            |
| `low_mid`  | 250 – 500 Hz  | 11                | Low mids, male vocals        |
| `mid`      | 500 – 2 000 Hz| 71                | Core melody, snare body      |
| `presence` | 2 – 6 kHz     | 186               | Vocals, attack transients    |
| `air`      | 6 kHz – Nyq.  | 744               | Cymbals, brightness, hiss    |

## Onset Strength

Each frame, `FftAnalyzer` computes one `onset_strength: f32` scalar via
**half-wave rectified log-magnitude spectral flux**:

```
flux[i] = max(0,  log(1 + C·mag_new[i])
             −    log(1 + C·mag_old[i]))
```

Two design choices:

- **Log compression** (`C = 1000`) — maps magnitudes onto a perceptual scale.
  Without it, the flux is dominated by absolute loudness rather than relative
  energy arrivals; quiet passages produce near-zero onset even on a sharp attack.
- **Bass weighting (3×)** for bins below 250 Hz — kick-drum transients are the
  dominant beat signal in electronic music. Equal-weight flux dilutes the kick
  with hi-hats, reverb tails, and noise.

The per-bin contributions are summed, divided by the number of bins, scaled by
`2`, and clamped to `[0, 1]`:

```
onset_strength = clamp(sum(flux) / n_bins × 2, 0, 1)
```

The beat tracker consumes this scalar; it has no knowledge of bins or
magnitudes. See [`beat/beat.md`](../beat/beat.md) for how the scalar feeds the
comb filter and PLL.

## Level Mapping (linear → 0–1)

Each band energy (a linear amplitude) is mapped to a `0.0–1.0` float on a
logarithmic (dB, decibel) scale:

```
dB    = 20 · log10(amplitude)
level = clamp((dB + 80) / 80, 0, 1)
```

The floor is **−80 dB** rather than the more common −60 dB, because high-
frequency bands (presence, air) are naturally quieter and would perpetually
clip to 0 with a shallower floor.

| linear amplitude | dB    | mapped level |
|-----------------|-------|--------------|
| 1e-10 (silence) | −200  | 0.0          |
| ~1e-4           | −80   | 0.0          |
| ~3.2e-3         | −50   | 0.375        |
| ~3.2e-2         | −30   | 0.625        |
| 1.0 (full scale)| 0     | 1.0          |
