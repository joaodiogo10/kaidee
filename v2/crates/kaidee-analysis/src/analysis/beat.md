# Beat Tracking Algorithm

Two-stage pipeline: **BPM estimation** (slow, ~1 Hz) drives a **PLL phase tracker**
(fast, runs every analysis frame at ~100 Hz).

```
FFT magnitudes ──▶ log-flux onset ──▶ onset ring buffer
                                            │
                                            ▼
                                     comb-filter scoring
                                    × tempo prior (1 Hz)
                                            │
                                            ▼
                                     BPM estimate ──▶ PLL phase tracker ──▶ beat_phase
                                                            ▲
                                        onset strength ─────┘
                                        (phase correction)
```

## Stage 1 — Onset Detection

Each FFT frame produces an **onset strength** value via *half-wave rectified
log-magnitude spectral flux*. For each frequency bin we compute

```
flux[i] = max(0, log(1 + C·mag_new[i]) − log(1 + C·mag_old[i]))
```

and sum across bins. Two tricks matter here:

- **Log compression** (`C = 1000`) maps raw magnitudes into a perceptual
  scale. Without it, the onset signal is dominated by absolute loudness —
  quiet passages produce weak onsets and loud passages saturate. Log-flux
  is roughly loudness-invariant, so the same pattern at −30 dB and 0 dB
  produces onsets of comparable strength.

- **Bass weighting (3×)** for bins below 250 Hz. Kick-drum transients are
  the dominant beat signal in electronic music; treating all bins equally
  dilutes them with hi-hats, reverb tails, and noise.

The resulting scalar is normalized to 0–1 and stored in a circular ring
buffer of `ONSET_HISTORY` frames (~8 s at 100 Hz).

## Stage 2 — BPM Estimation (comb filter + tempo prior, ~1 Hz update)

Once the onset buffer is full, BPM is re-estimated every ~1 s. The scoring
loop sweeps every lag `L` in the 60–200 BPM range and computes a
**comb-filter score**: how well does the onset signal align with a pulse
train at `L`, `2L`, `3L`?

```
corr(L) = (1/terms) · Σ_t  onset[t] · (onset[t+L] + onset[t+2L] + onset[t+3L])
```

This is the main upgrade over raw single-lag autocorrelation. Autocorrelation
scores `L` by how many pairs of onsets are `L` apart — but in a typical
four-on-the-floor pattern, onsets are *also* `2L` and `3L` apart, so half
and third tempos pick up almost the same score as the true tempo. The comb
filter reinforces genuine tempos across *multiple* periods while short-lag
noise and sparse accidental correlations don't stack the same way.

Per-lag normalization by `terms = n − 3L` keeps the comparison fair: longer
lags accumulate fewer pairs, and without the divisor their scores collapse.

### Tempo prior (octave disambiguation)

Even with the comb filter, period-L and period-2L signals score similarly —
both are genuine periodicities. To break the tie we multiply each candidate's
score by a **log-Gaussian tempo prior** centered on 120 BPM:

```
prior(bpm) = exp(−(log(bpm/120))² / (2·σ²))      σ = 0.4 log-octaves
```

This isn't a hard constraint — it just gently pulls scoring toward musically
plausible tempos. At the 60 BPM and 200 BPM edges the prior is ~0.5, so
a strong signal there can still win. The effect is to break octave ties
(70 vs 140, 80 vs 160) reliably in favor of the musically sensible answer.

### Change cap + slow EMA

Once the winning lag yields a candidate BPM:

```
clamped = clamp(candidate, bpm − 0.08·bpm, bpm + 0.08·bpm)
bpm     = 0.85·bpm + 0.15·clamped
```

The 8% cap prevents a single noisy frame from throwing the estimate across
the range; the 0.85/0.15 EMA smooths remaining jitter while still letting
real tempo drift track over a few seconds.

## Stage 3 — PLL Phase Tracker (100 Hz)

A free-running phase accumulator `phase ∈ [0, 1)` advances by
`bpm/60 / analysis_rate` each frame (0 = beat, 1 = next beat).

On strong onsets (strength > 0.3) the phase is nudged toward the nearest
beat boundary:

| phase    | interpretation              | correction            |
|----------|-----------------------------|-----------------------|
| `< 0.5`  | just after a beat           | pull back toward 0    |
| `> 0.5`  | approaching the next beat   | push forward toward 1 |

Correction gain is `0.12 × onset_strength`, kept small so the PLL tracks
without snapping. Phase wrapping always uses `rem_euclid(1.0)`, which
returns a non-negative result for negative inputs — the fractional-part
method does not, and would require a manual fixup after each correction.

## Tuning Parameters

| Parameter             | Value | Effect                                              |
|-----------------------|-------|-----------------------------------------------------|
| `ONSET_HISTORY`       | 800   | ~8 s of onset history for comb scoring             |
| Log compression `C`   | 1000  | Dynamic-range compression in spectral flux         |
| Bass bin weight       | 3×    | Emphasis on kick-drum band (<250 Hz)               |
| Onset scale factor    | 2.0   | Maps post-log flux to useful 0–1 range             |
| Comb harmonics        | 3     | Pulse-train terms in BPM scoring (`L, 2L, 3L`)     |
| Prior center          | 120   | BPM center of tempo prior                          |
| Prior σ               | 0.4   | Tempo-prior width (log-octaves)                    |
| PLL threshold         | 0.3   | Minimum onset strength for phase correction        |
| PLL gain              | 0.12  | Fraction of phase error corrected per strong onset |
| BPM change cap        | 8%    | Max per-update BPM shift                           |
| BPM EMA               | 0.15  | Weight on new BPM estimate per update              |
