# Dynamics

Derives two overall loudness/level features from the magnitude spectrum:
**RMS** (root mean square) energy and **peak** magnitude.

Both are computed from the FFT magnitude spectrum rather than the raw
time-domain signal. This is a deliberate trade-off: the spectral domain is
already available from the FFT step, so there is no extra computation, and
spectral RMS correlates well with perceived loudness for music signals.

## RMS

```
rms = √( (1/N) · Σ magnitude[i]² )
```

The mean of squared magnitudes, then square-rooted. This is the spectral
equivalent of time-domain RMS — it captures average signal power across the
full audible range. Loud, dense mixes produce high RMS; sparse or quiet
signals produce low RMS.

## Peak

```
peak = max( magnitude[0], magnitude[1], …, magnitude[N−1] )
```

The single loudest bin in the spectrum. Useful for detecting sharp transients
(kick hits, claps) that briefly spike without raising the average level much.
A high peak with low RMS indicates a percussive transient; high peak with high
RMS indicates a dense, sustained sound.

## Level Mapping

Both values are passed through the same dB (decibel) → 0–1 mapping used in
spectral band energies (floor −80 dB, ceiling 0 dB). See `spectral.md` for
the formula.

## Future

- **LUFS** (loudness units relative to full scale) — perceptual loudness
  weighting (K-weighting filter) for more broadcast-accurate level metering.
- **Crest factor** — peak / RMS ratio; quantifies how "punchy" a signal is
  independent of its level.
