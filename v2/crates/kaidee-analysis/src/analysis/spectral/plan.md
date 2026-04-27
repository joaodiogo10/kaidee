# Spectral Analyzer Roadmap

Alternative strategies for the [`SpectralAnalyzer`] trait. Each strategy is a
drop-in replacement — swap via [`Pipeline::with_analyzer`] at construction time.

## Architecture principle: spectral analysis and beat tracking are co-designed

The beat tracker is a temporal pattern detector. Its only job is to find
periodicity in a stream of onset strengths. It never needs to understand
frequency bins — that is the spectral layer's responsibility. Each strategy
exposes `onset_strength() -> f32`, computed using its own representation.
Other downstream consumers that need frequency-specific onset signals derive
them from consecutive `BandEnergies` frames independently.

**Swapping strategies intentionally changes beat tracker performance.** This is
correct behaviour, not a side-effect. An IIR filterbank with ~10 ms latency
produces sharper onset signals → more precise beat phase. A CQT with 10× more
bass bins produces stronger onset peaks on kick drums → faster BPM convergence.

**The beat tracker's tunable constants are strategy-dependent.** The onset
threshold (`0.3`), PLL (phase-locked loop) gain (`0.12`), and BPM smoothing
weights (`0.85/0.15`) were calibrated against `FftAnalyzer` output. Different
strategies produce onset signals with different noise floors, transient
amplitudes, and dynamic ranges. These constants should be documented per
strategy in each strategy's `.md` file once empirically validated.

## Current state

```
spectral/
├── mod.rs       — SpectralAnalyzer trait + BandEnergies + onset_strength  ✓ done
├── fft.rs       — FftAnalyzer + onset computation + onset tests           ✓ done
├── fft.md
├── iir.rs       — IirFilterbank (31-band 1/3-octave biquad filterbank)    ✓ done
├── iir.md
beat.rs          — BeatTracker: pair-product comb, parabolic interp, PLL   ✓ done
pipeline.rs      — routes onset_strength() into tracker.update()           ✓ done
```

`BeatTracker` receives a scalar `onset_strength` per frame — no spectral
knowledge. Algorithm: pair-product comb filter → log-Gaussian tempo prior
(center 150 BPM) → parabolic peak interpolation → EMA smoothing → PLL phase
tracker. All click-track tests converge within ±0.5 BPM.

## Target layout

```
spectral/
├── mod.rs          — trait + BandEnergies                  ✓ done
├── fft.rs          — FftAnalyzer                           ✓ done
├── fft.md          —                                       ✓ done
├── iir.rs          — IirFilterbank                         ✓ done
├── iir.md          —                                       ✓ done
├── multi_fft.rs    — MultiFftAnalyzer                      step 3
├── multi_fft.md
├── cqt.rs          — CqtAnalyzer                           step 4
└── cqt.md
```

---

## Step 1 — Decouple BeatTracker from spectral representation ✓ done

`BeatTracker.update` now accepts `onset: f32` instead of `magnitudes: &[f32]`.
`spectral_flux` moved into `FftAnalyzer::compute_onset`, called during
`process()` and stored as a field. `BeatTracker` no longer holds
`prev_magnitudes`, `n_bass_bins`, or any frequency-bin knowledge.
`onset_strength()` added to `SpectralAnalyzer` trait. Two new tests added to
`fft.rs`: `onset_fires_on_attack` and `onset_decays_on_steady_state`.
12 tests pass.

---

## Step 2 — IirFilterbank ✓ done

**Why first.** Lowest latency (~10 ms vs current ~46 ms), no spectral leakage,
no new crate dependencies.

### Algorithm

A bank of 31 biquad (bi-quadratic) bandpass filters at ISO 61260 1/3-octave
centre frequencies (20 Hz to ~20 kHz). Each filter runs continuously on every
incoming sample — no block processing. `process()` is a snapshot of accumulated
RMS (root mean square) energy since the last call.

Centre frequencies follow the 1/3-octave sequence: each step multiplies by
`2^(1/3) ≈ 1.2599`, starting at ~20 Hz.

Q factor for 1/3-octave constant-bandwidth: `Q ≈ 4.318`.

### Biquad design (bilinear-transform bandpass)

For each centre frequency `fc`:

```
ω₀ = 2π · fc / sample_rate
α  = sin(ω₀) / (2 · Q)

b0 =  α / a0        b1 =  0       b2 = -α / a0
a1 = -2·cos(ω₀)/a0  a2 = (1-α)/a0    where a0 = 1 + α
```

### Filter topology: Transposed Direct Form II (TDF-II)

Two state variables per filter (`s1`, `s2`). Numerically better than Direct
Form I (4 state variables) for f32 accumulation of small values:

```
y  = b0·x + s1
s1 = b1·x - a1·y + s2
s2 = b2·x - a2·y
```

### Data flow

```
push_sample(x)
  for each of 31 filters:
    y = TDF-II(x)
    accumulator[i] += y²
    sample_count[i] += 1

process()
  for each band:
    energy[i]      = √(accumulator[i] / sample_count[i])  ← per-band RMS
    onset[i]       = (energy[i] - prev_energy[i]).max(0.0) ← positive flux
    prev_energy[i] = energy[i]
    accumulator[i] = 0
    sample_count[i] = 0
  aggregate 31 narrow bands → 6 canonical BandEnergies + OnsetFrame
```

### Trait impl notes

- `n_bins()` → 31
- `magnitudes()` → 31-element per-band RMS slice
- `band_energies()` → aggregate ISO bands into the 6 canonical ranges
- `onset_strength()` → bass-weighted positive RMS delta across ISO bands

### Dynamics note

`dynamics::compute` runs on `magnitudes()`. With 31 values instead of 1024 the
RMS calibration from `to_01` shifts. A future improvement is a parallel
time-domain RMS accumulator in `push_sample` for dynamics, bypassing
`magnitudes()` entirely.

### Tests

| Test | What it checks |
|------|----------------|
| `silence_all_bands_zero` | All-zero input → all outputs 0.0 |
| `sine_NNNhz_activates_*` | One test per canonical band, stricter thresholds than FFT (no leakage) |
| `onset_fires_on_attack` | Sudden sine burst → `onset_strength` > 0 |
| `onset_silent_on_steady_state` | Sustained sine → `onset_strength` decays to ~0 after a few frames |
| `accumulator_resets_between_calls` | Second `process()` sees new data, not stale |
| `magnitudes_length_is_n_bands` | `magnitudes().len() == n_bins()` |
| `filter_poles_stable` | All poles inside unit circle (constructor sanity) |
| `click_track_120bpm_converges` | Beat tracker regression via `Pipeline::with_analyzer` |

---

## Step 3 — MultiFftAnalyzer *(priority 2)*

**Why.** Addresses the FFT time-frequency trade-off with no new dependencies.
Low frequencies get fine frequency resolution; high frequencies get fine time
resolution.

### Algorithm

Three independent FFT (fast Fourier transform) instances run in parallel, each
tuned to a frequency region:

| Instance | FFT size | Hz/bin   | Latency  | Region          |
|----------|----------|----------|----------|-----------------|
| Low      | 8192     | ~5.4 Hz  | ~186 ms  | sub-bass + bass |
| Mid      | 2048     | ~21.5 Hz | ~46 ms   | low-mid + mid   |
| High     | 512      | ~86 Hz   | ~12 ms   | presence + air  |

`push_sample` feeds every sample into all three circular buffers. `process()`
runs all three FFTs and builds a concatenated magnitude slice:

```
magnitudes = [low_mags[0..bass_end], mid_mags[bass_end..mid_end], high_mags[mid_end..]]
```

Index offsets are computed once at construction. The concatenated slice has
non-uniform Hz/bin spacing — fine for beat tracking (kick transients are
well-resolved in the low sub-array) and band energy computation (each band reads
from the correct sub-array).

### Onset computation

Bass-weighted spectral flux from the low sub-array (5.4 Hz/bin → better kick
resolution than `FftAnalyzer`). The low sub-array's finer bin resolution means
kick drum transients produce a stronger, cleaner onset signal.

### Trait impl notes

- `n_bins()` → length of concatenated slice
- `band_energies()` → Euclidean-norm accumulation per region of the concatenated slice
- `onset_strength()` → bass-weighted log flux from the low sub-array

### Tests

| Test | What it checks |
|------|----------------|
| `silence_all_bands_zero` | All-zero input → 0.0 |
| `bass_sine_fine_resolution` | 40 Hz sine confined to fewer bins than FftAnalyzer |
| `high_freq_fast_attack` | 10 kHz sine detected after 512 samples (one high-FFT window) |
| `onset_fires_on_attack` | Sudden bass burst → `onset_strength` > 0 |
| `magnitudes_length_correct` | Concatenated slice length matches `n_bins()` |
| `click_track_120bpm_converges` | Beat tracker regression |

---

## Step 4 — CqtAnalyzer *(priority 3)*

**Why.** Gold-standard for music analysis. Each bin maps to a musical interval
(semitone). Sub-bass and bass have 10× more bins than FftAnalyzer.

### Algorithm

Constant-Q Transform: bin centre frequencies are logarithmically spaced and
each bin's effective window length is inversely proportional to its frequency.
Result: equal musical resolution across all octaves.

Parameters: **12 bins/octave**, Q ≈ 17, spanning 20 Hz–20 kHz → ~120 bins.

### Kernel precomputation (Brown & Puckette method)

For each CQT bin `k` with centre frequency `f_k`:

1. Compute window length: `N_k = Q · sample_rate / f_k`
2. Build complex Hann-windowed exponential kernel of length `N_k`:
   `h_k[n] = (1/N_k) · hann(n/N_k) · exp(-j·2π·Q·n/N_k)`
3. Zero-pad to `FFT_SIZE = 65536`, FFT to get spectral kernel `H_k`
4. Sparsify: keep entries where `|H_k[i]| > 0.005 · max(|H_k|)`
   → typically 30–100 non-zero entries per bin

### Per-frame computation

```
process()
  1. copy circular window → FFT buffer, apply Hann, run 65536-pt FFT
  2. for each CQT bin k:
       cqt[k] = Σ fft_out[i] · conj(H_k[i])   (sparse multiply)
       magnitudes[k] = |cqt[k]|
  3. compute OnsetFrame: positive log-flux per frequency band,
     summed over bins whose centre_frequencies[k] falls in that band's range
```

CPU cost at 100 Hz: one 65536-pt FFT (~1 ms) + 120 × ~60 complex MACs
(negligible).

### Practical constraint

The 20 Hz bin requires `N_0 = 17 · 44100 / 20 ≈ 37 470` samples. This is
the outer FFT size floor — hence `FFT_SIZE = 65536` (next power of two). The
large FFT is computed once per hop; the kernel is precomputed at construction.

### Trait impl notes

- `n_bins()` → 120 (or chosen bins-per-octave × octaves)
- `band_energies()` → iterate `centre_frequencies` to bucket bins into the 6 ranges
- `onset_strength()` → bass-weighted positive log-flux across bins with centre < 250 Hz

### Tests

| Test | What it checks |
|------|----------------|
| `silence_all_bands_zero` | All-zero input → 0.0 |
| `sine_activates_correct_bin` | 440 Hz → peak at bin nearest A4 |
| `bins_per_octave_ratio` | Consecutive centres at ~semitone ratio (2^(1/12) ≈ 1.0595) |
| `bass_sub_bass_separation` | 40 Hz sine: sub_bass active, bass near zero (tighter than FFT) |
| `onset_fires_on_attack` | 40 Hz sine burst → `onset_strength` > 0 |
| `kernel_non_empty` | Every kernel vector has at least one entry |
| `click_track_120bpm_converges` | Beat tracker regression |

---

## Dependency graph

```
Step 1  (OnsetFrame + BeatTracker decoupling)   ← must be first
  └─ Step 2  (IIR)
  └─ Step 3  (MultiFft)
  └─ Step 4  (CQT)
```

No new crate dependencies for any strategy — `rustfft` (already in
`Cargo.toml`) covers all FFT needs; IIR is pure arithmetic.

## Strategy comparison

| Property            | FftAnalyzer | IirFilterbank | MultiFftAnalyzer | CqtAnalyzer |
|---------------------|-------------|---------------|------------------|-------------|
| Latency             | ~46 ms      | ~10 ms        | ~10 ms (high)    | ~46 ms      |
| Bass Hz/bin         | 21.5 Hz     | n/a (bands)   | 5.4 Hz           | log (Q=17)  |
| Spectral leakage    | Present     | None          | Present          | Minimal     |
| CPU (relative)      | 1×          | ~0.5×         | ~3×              | ~5×         |
| New dependencies    | —           | —             | —                | —           |
| Best for            | General     | Transients    | Kick/bass detail | Pitch/chords|
