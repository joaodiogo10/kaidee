# IirFilterbank — ISO 61260 1/3-Octave IIR Filterbank

Processes audio sample-by-sample through 31 biquad bandpass filters at ISO
61260 standard centre frequencies. Unlike block-based FFT analysis, there is no
window accumulation — `process()` is a snapshot of energy accumulated since the
last call.

```
push_sample(x)
  for each of 31 ISO bands:
    y = Biquad(x)          ← runs at full sample rate
    accumulator += y²
    count += 1

process()   ← called once per hop (~100 Hz)
  for each band:
    band_rms = √(accumulator / count)
    onset_delta = (log(1 + C·band_rms) − log(1 + C·prev_rms)).max(0)
    accumulator = 0,  count = 0
  onset = Σ weight·delta / N_BANDS   (bass bands weighted 3×)
  aggregate 31 ISO bands → 6 canonical BandEnergies
```

## Trade-offs

| Property              | Value                                                         |
|-----------------------|---------------------------------------------------------------|
| Effective latency     | ~10 ms (one hop, no window accumulation)                      |
| Inter-band rejection  | Poor with 2nd-order filters: ~-27 dB at 3 octaves away       |
| Frequency resolution  | Fixed ISO 1/3-octave centres (cannot be changed)              |
| Band boundary precision | Approximate — 1/3-octave skirts span ~26% BW               |
| CPU cost              | O(N_BANDS) per sample — lighter than FFT                      |

**Band isolation vs FFT.** A 2nd-order biquad rolls off at -40 dB/decade per
side. A 40 Hz sine produces ~-27 dB at the 500 Hz filter, mapping to ~0.66 on
the 0–1 scale. By contrast, FFT with a Hann window has exponentially decaying
sidelobes and produces exact 0.0 at mid for a 40 Hz input. The IIR advantage
is **latency**, not isolation. The target band is always the dominant one;
non-target bands are non-zero but lower. For better isolation, use cascaded
biquads (4th or 6th order per band) at the cost of more state per filter.

## ISO 61260 Centre Frequencies

31 bands from 20 Hz to 20 kHz in 1/3-octave steps (each ×2^(1/3) ≈ 1.2599):

| Canonical band | ISO centre frequencies (Hz)                            | Count |
|----------------|--------------------------------------------------------|-------|
| sub_bass       | 20, 25, 31.5, 40, 50                                  | 5     |
| bass           | 63, 80, 100, 125, 160, 200                            | 6     |
| low_mid        | 250, 315, 400                                         | 3     |
| mid            | 500, 630, 800, 1 000, 1 250, 1 600                   | 6     |
| presence       | 2 000, 2 500, 3 150, 4 000, 5 000                    | 5     |
| air            | 6 300, 8 000, 10 000, 12 500, 16 000, 20 000         | 6     |

## Biquad Design (Bilinear-Transform Bandpass)

For each centre frequency `fc`, Q = 4.318 (constant-Q for 1/3-octave bandwidth,
derived from Q = 1 / (2^(1/6) − 2^(−1/6))):

```
ω₀ = 2π · fc / sample_rate
α  = sin(ω₀) / (2 · Q)

b0 =  α / a0     b1 = 0     b2 = −α / a0
a1 = −2·cos(ω₀) / a0        a2 = (1−α) / a0     where a0 = 1 + α
```

`b1 = 0` is exact for a symmetric bandpass filter — the TDF-II update drops
the b1 term entirely.

## Filter Topology: Transposed Direct Form II (TDF-II)

Two state variables per filter. Better f32 numerical stability than Direct
Form I (four state variables) when accumulating small amplitudes.

```
y  = b0·x + s1
s1 = −a1·y + s2      ← b1·x dropped (b1 = 0)
s2 =  b2·x − a2·y
```

## Band Boundary Approximation

The canonical band boundaries (60 Hz, 250 Hz, 500 Hz, 2 000 Hz, 6 000 Hz) do
not align exactly with ISO band centres. A 250 Hz filter spans roughly
224–281 Hz — it straddles the bass/low-mid boundary. Assignment is by centre
frequency: the 250 Hz ISO band is counted as `low_mid`.

This is inherent to 1/3-octave analysis and has no meaningful impact on
visual reactivity. It is distinct from FFT spectral leakage — adjacent
canonical bands do not bleed into each other beyond one ISO band width.

## Beat Tracker Constants (tuned against FftAnalyzer)

The onset threshold (`0.3`), PLL (phase-locked loop) gain (`0.12`), and BPM
EMA (exponential moving average) smoothing (`0.65 / 0.35`) in `beat.rs` were
calibrated against `FftAnalyzer` output. IIR onset signals have lower latency
(~10 ms vs ~46 ms) and sharper transient edges — the PLL may converge faster
and the onset threshold may need lowering if phase jitter increases. Validate
with `click_track_120bpm_converges` and real audio fixtures.
