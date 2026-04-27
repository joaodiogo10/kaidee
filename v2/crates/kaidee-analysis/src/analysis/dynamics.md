# Dynamics

Derives two overall loudness/level features from the magnitude spectrum:
**RMS** (root mean square) energy and **peak** magnitude.

Both are computed from the FFT magnitude spectrum rather than the raw
time-domain signal. This is a deliberate trade-off: the spectral domain is
already available from the analysis step, so there is no extra computation, and
spectral RMS correlates well with perceived loudness for music signals.

## RMS

```
rms = √( (1/N) · Σ magnitude[i]² )
```

The mean of squared magnitudes, then square-rooted. This is the spectral
equivalent of time-domain RMS — it captures average signal power across the
full audible range. Loud, dense mixes produce high RMS; sparse or quiet
signals produce low RMS.

**Note:** spectral RMS over 1 024 FFT bins is much lower than time-domain RMS
for pure tones — a full-scale 440 Hz sine activates only ~1 bin out of 1 024,
so the spectral mean-square is 1/1024th of what a broadband signal at the same
amplitude would produce. The mapped output for a full-scale sine is ~0.55, not
~1.0. This is expected and documented in `dynamics.rs`.

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
spectral band energies (floor −80 dB, ceiling 0 dB). See `fft.md` for the
formula and motivation.

## Future

- **LUFS** (loudness units relative to full scale) — perceptual loudness
  weighting (K-weighting filter) for more broadcast-accurate level metering.
- **Crest factor** — peak / RMS ratio; quantifies how "punchy" a signal is
  independent of its level.
- **Time-domain RMS** — a parallel accumulator in `push_sample` would give
  true sample-level RMS, avoiding the spectral-bin dilution described above.
