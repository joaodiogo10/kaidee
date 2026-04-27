# Beat Tracking Algorithm

Two-stage pipeline: **BPM estimation** (slow, every `update_interval`) drives
a **PLL (phase-locked loop) phase tracker** (fast, one step per analysis
frame — i.e. at [`analysis_rate`](../analysis.md#analysis_rate)). See
"Derived quantities" below for the values these resolve to at the current
config.

```
onset_strength (scalar)  ──▶  onset ring buffer (1200 frames ≈ 12 s)
                                        │
                                        ▼
                               pair-product comb scoring
                              × log-Gaussian tempo prior
                               (every 0.5 s, after 4 s warmup)
                                        │
                              parabolic peak interpolation
                                        │
                                        ▼
                               BPM estimate ──▶ PLL phase tracker ──▶ beat_phase
                                                       ▲
                                   onset_strength ─────┘
                                   (phase correction)
```

The beat tracker receives a single `onset_strength: f32` per frame from the
spectral analyzer — it has no knowledge of frequency bins or magnitudes. Onset
computation is the spectral layer's responsibility.

## Derived quantities

Most of the buffer sizes, lag bounds, and update cadences in the algorithm
are *derived* from a handful of base constants. The frame counts cited
throughout this document are evaluations of those formulas at the current
configuration — change any base constant and the derived values recompute.

**Base constants** (all referenced by name in the text below). Plus the
global [`analysis_rate`](../analysis.md#analysis_rate) from the root doc,
which appears in every derived-quantity formula in this document.

| Constant             | Value     | Source                                 |
|----------------------|-----------|----------------------------------------|
| `MIN_BPM`, `MAX_BPM` | 20, 300   | `beat.rs` — fixed operational range    |
| `N_PAIRS`            | 3         | `beat.rs`                              |
| `update_interval`    | 0.5 s     | `beat.rs` (`BPM_UPDATE_INTERVAL_SECS`) |
| `WARMUP_MIN_BPM`     | 60 BPM    | `beat.rs` — fastest tempo eligible at warmup |

**Derived quantities:**

| Symbol               | Formula                                          | At current config   |
|----------------------|--------------------------------------------------|---------------------|
| `min_lag`            | `analysis_rate · 60 / MAX_BPM`                   | 20 frames (0.2 s)   |
| `max_lag`            | `analysis_rate · 60 / MIN_BPM`                   | 300 frames (3.0 s)  |
| `history_len`        | `(N_PAIRS + 1) · max_lag`                        | 1200 frames (12 s)  |
| `update_hop`         | `analysis_rate · update_interval`                | 50 frames (0.5 s)   |
| `WARMUP_MIN_FRAMES`  | `(N_PAIRS + 1) · analysis_rate · 60 / WARMUP_MIN_BPM` | 400 frames (4 s) |

## Stage 1 — Onset Strength (spectral layer handoff)

Each analysis frame the spectral analyzer produces one `onset_strength: f32`
scalar in the range 0–1. The beat tracker stores it in a circular ring buffer
of size `history_len` (see Derived quantities) — one pair-set at the slowest
tempo in the search range.

How `onset_strength` is computed is the spectral layer's responsibility — see
[`spectral/fft.md`](../spectral/fft.md) for the algorithm. The beat tracker
has no knowledge of frequency bins or magnitudes; swapping the spectral
strategy changes the onset signal's characteristics and may require re-tuning
the constants below.

## Stage 2 — BPM Estimation (every `update_interval`)

BPM is re-estimated once `WARMUP_MIN_FRAMES` onsets have accumulated, then
every `update_hop` frames (see Derived quantities). The cadence is a
cost/latency trade-off: more frequent updates re-scan a barely-changed
buffer; less frequent adds latency when the tempo genuinely shifts.
`update_interval = 0.5 s` is short enough that the 20% change cap (see
"BPM update" below) can still traverse a tempo change within a few seconds.

Warmup is **decoupled from the buffer size**: the ring buffer holds
`history_len` frames (sized for `MIN_BPM`), but BPM estimation starts as
soon as `WARMUP_MIN_FRAMES` onsets exist. The comb's search range is
restricted by a **dynamic lag cap**:

```
lag_cap = min(max_lag, available_frames / (N_PAIRS + 1))
```

The cap comes directly from the pair-product sum needing `N_PAIRS + 1`
equally-spaced samples to form `N_PAIRS` pairs. The warmup threshold is
derived from this: `WARMUP_MIN_FRAMES` is exactly the number of frames
needed for `lag_cap` to reach the lag of `WARMUP_MIN_BPM` — so by
construction, tempos ≥ `WARMUP_MIN_BPM` are eligible at the first
estimate. As the ring continues to fill, the cap rises linearly until
`available = history_len`, at which point the full `MIN_BPM..=MAX_BPM`
range is searched. Fast tempos lock in at the warmup boundary; slow
tempos emerge as the ring fills. This reflects a physical floor — you
cannot detect a tempo whose period does not fit `N_PAIRS + 1` times into
the window.

### Pair-product comb filter

For each candidate lag `L` in `min_lag..=lag_cap` the comb score is:

```
corr(L) = (1/terms) · Σ_t Σ_k  onset[t + k·L] · onset[t + (k+1)·L]
                                                  k = 0, 1, 2   (N_PAIRS = 3)
```

Each pair in the inner sum is only non-zero when **both** frames are beats.
That structural zero kills one class of harmonic locks but not the other:

| Candidate lag    | Sum-of-harmonics    | Pair-product           |
|------------------|---------------------|------------------------|
| `L/2` (faster than true — the tempo is doubled) | half the positions are beats → spurious half-score | paired with silence between beats → **0** ✓ |
| `L`  (true period) | full score          | full score             |
| `2L` (slower than true — the tempo is halved)  | all positions land on beats → full spurious score | pairs are `(beat, beat)` spaced further apart → **full spurious score** ✗ |

Pair-product's win is one-sided: it suppresses candidates *faster* than the
true period (the factor between beats is silence), but not candidates
*slower* than it (both factors still land on beats, just further apart).
Breaking slow-harmonic ties (`L` vs `2L` vs `3L`) is the prior's job.

Per-lag normalization by `terms = n − N_PAIRS·L` keeps scores comparable
across lags of different lengths. A side effect is that on a pure periodic
signal every slow harmonic (`L`, `2L`, `3L`, …) scores **identically** under
pair-product — the prior and the fast-harmonic probe are the only
mechanisms that pick one over the others.

### Log-Gaussian tempo prior (octave disambiguation)

The prior multiplies each candidate's comb score to break remaining ties:

```
prior(bpm) = exp(−(log(bpm / 150))² / (2 · 0.5²))
```

Center at **150 BPM** (not 120). With center = 120, `prior(88) > prior(175)`,
causing the tracker to prefer the 2× sub-harmonic for fast tempos. At 150 BPM
the ordering reverses (`prior(175) = 0.95 > prior(88) = 0.82`) while tempos
in the 60–140 BPM range remain within ~2σ. Note σ is in natural-log units
of the BPM ratio: σ = 0.5 corresponds to a factor of `e^0.5 ≈ 1.65`, i.e.
about 0.72 octaves.

The prior alone only keeps the true tempo favoured up to the crossover
`B*` where `prior(B*) = prior(B*/2)`. Solving `log(B/150)² = log(B/300)²`
gives `B* = √(150·300) ≈ 212 BPM` — above this, the half-tempo scores
higher. The **fast-harmonic probe** (next) corrects this structurally
without retuning the prior.

### Fast-harmonic probe

On pair-product the true tempo and every slow harmonic (`L`, `2L`, `3L`, …)
tie on normalized corr (the equalization discussed above), so the score
winner is decided by the prior — which tips the wrong way above ~212 BPM.
After the main scan picks `best_lag` by score, the tracker iteratively
halves:

```
while best_lag / 2 ≥ min_lag
  and corr(best_lag / 2) ≥ 0.95 · corr(best_lag):
    best_lag = best_lag / 2
```

The comparison is on **raw corr**, not score, so it ignores the prior
entirely. The loop is safe because of the asymmetry in the pair-product
table above: a lag *shorter* than the signal's true period scores a
structural zero, so `corr(best_lag / 2)` only survives the threshold when
the signal genuinely contains the shorter period. Real slow tempos (e.g.
60 BPM kick) get `corr = 0` at half their lag and don't flip.

**Why 0.95 (and not 0.7).** Real music routinely has on-beat
subdivisions — 8th-note hi-hats on a 4-on-the-floor kick, snares sharing
energy with ghost notes, etc. Model the worst case: a perfectly periodic
click at period `L` with main beats at strength 1 and subdivisions at
strength `s` on the half-beat (period `L/2`, offset `L/2`). Pair-product
sums then give:

```
corr(L)   ∝ 3·(1 + s²)   // main beats pair with main beats; subs with subs
corr(L/2) ∝ 6s           // every pair is (main, sub) or (sub, main)

ratio = corr(L/2) / corr(L) = 2s / (1 + s²)
```

| Subdivision strength `s` | `corr(L/2) / corr(L)` |
|--------------------------|-----------------------|
| 0.3 (soft hats)          | ≈ 0.55                |
| 0.5 (typical hats)       | ≈ 0.80                |
| 0.7 (loud hats)          | ≈ 0.94                |
| 1.0 (true doubled tempo) | 1.00                  |

A threshold of 0.7 would wrongly flip a 120 BPM track with typical hats
(ratio ≈ 0.80) to 240 BPM. 0.95 rejects everything up to about `s ≈ 0.72`
in the idealised case, and real music has enough per-beat variation that
the effective ratio sits well below this ceiling. Click-track-perfect
ties (ratio 1.0) still pass, which is the case the probe exists to
handle. The cost is that real 250–300 BPM material with strong
beat-to-beat strength variation won't flip; that trade-off favours
false-negative doubling (still musical) over false-positive doubling
(audibly wrong).

The probe runs at most `log₂(max_lag / min_lag)` iterations (~4 at the
current config). Each iteration is one comb scan at a single lag —
negligible cost next to the main scan.

### Parabolic interpolation

Integer lags introduce discretization error
(e.g. lag = 34 → 176.47 BPM, not 175.0). After finding the best integer lag,
a parabola is fit through the comb scores at `best_lag − 1`, `best_lag`,
`best_lag + 1`:

```
δ = 0.5 · (s₋ − s₊) / (s₋ − 2·s₀ + s₊)
fractional_lag = best_lag + clamp(δ, −1, 1)
```

The refinement removes the quantization bias and brings the converged estimate
within ±0.5 BPM of the true tempo.

### BPM update

```
candidate = analysis_rate × 60 / fractional_lag
clamped   = clamp(candidate, bpm − 0.20·bpm, bpm + 0.20·bpm)
bpm       = 0.65·bpm + 0.35·clamped
```

The 20% change cap bounds per-update BPM shift geometrically: traversing
the full `MIN_BPM..=MAX_BPM` range takes
`ln(MAX_BPM / MIN_BPM) / ln(1.20)` update cycles — ≈ 15 cycles (~7.5 s)
at the current config. In practice the initial guess of 120 BPM is
already close to most real tempos, so convergence is faster —
e.g. 120 → 175 BPM is ~2 cycles (~1 s) of clamped climb. The EMA
(exponential moving average) with weight 0.35 gives a time constant of
~2.9 updates (~1.5 s), settling within ±0.5 BPM in ~3 s once the
interpolated estimate stabilizes.

## Stage 3 — PLL Phase Tracker (per frame, at `analysis_rate`)

A free-running phase accumulator `phase ∈ [0, 1)` advances by
`bpm / (60 · analysis_rate)` each frame (0 = beat, 1 = next beat).

On strong onsets (strength > 0.3) the phase is nudged toward the nearest
beat boundary:

| phase    | interpretation              | correction               |
|----------|-----------------------------|--------------------------|
| `< 0.5`  | just after a beat           | pull back toward 0       |
| `> 0.5`  | approaching the next beat   | push forward toward 1    |

Correction: `phase −= err × 0.12 × onset_strength`

Gain 0.12 is kept small so the PLL tracks without snapping. Phase wrapping
uses `rem_euclid(1.0)`, which returns a non-negative result for negative
inputs — the fractional-part method does not.

## Tuning Parameters

Base constants only — derived quantities (`history_len`, `update_hop`,
`WARMUP_MIN_FRAMES`, `min_lag`, `max_lag`) are listed in "Derived
quantities" above.

| Parameter             | Value    | Effect                                            |
|-----------------------|----------|---------------------------------------------------|
| `MIN_BPM` / `MAX_BPM` | 20 / 300 | Operational BPM range (fixed, not tunable)        |
| `N_PAIRS`             | 3        | Pair-product terms per lag (pairs at L, 2L, 3L)   |
| `WARMUP_MIN_BPM`      | 60 BPM   | Fastest tempo guaranteed eligible at first estimate |
| `update_interval`     | 0.5 s    | Re-estimation cadence                             |
| Prior center          | 150 BPM  | Center of log-Gaussian tempo prior                |
| Prior σ               | 0.5      | Width of prior (natural-log units of BPM ratio)   |
| BPM change cap        | 20%      | Max per-update BPM shift                          |
| BPM EMA weight        | 0.35     | Weight on new estimate; τ ≈ 1.5 s                 |
| PLL threshold         | 0.3      | Minimum onset strength for phase correction       |
| PLL gain              | 0.12     | Fraction of phase error corrected per strong onset |

## Known Weaknesses & Possible Improvements

The current algorithm is tuned for 4-on-the-floor electronic music at 120–180
BPM. Tracks that drift outside that envelope expose specific failure modes.
Each item below names the failure, the mechanism behind it, and a targeted
fix.

### 1. Syncopated basslines capture the PLL on a harmonic ratio

**Observed on** the ACIDSEX test track (true tempo 126 BPM): the tracker
drifts toward ~168 BPM (a 4/3 harmonic) during sections where the 303
bassline plays a triplet-feel pattern. The sustained bass notes produce
strong flux on off-beats and shift the dominant periodicity.

**Mechanism.** The pair-product comb is good at rejecting integer sub-harmonics
(2×, 3×) but has no structural defence against *non-integer* ratios like 4/3
or 3/2 — they produce genuine pair coincidences at a different lag.

**Fix — sub-bass-weighted onsets (spectral layer).** Weight only the
20–100 Hz bins heavily (e.g. 5×) and *down*-weight 100–250 Hz so sustained
synth-bass content contributes less. Kick drums stay dominant; bassline
syncopation fades.

**Fix — fundamental-period preference (beat tracker).** After picking the
best comb lag `L*`, probe `L*·3/4`, `L*·2/3`. If a shorter lag exists whose
comb score is ≥ 0.7 × score(L*), prefer it. This adds one structural test
for the dominant harmonic-ratio error case.

### 2. Extreme tempos are under-favoured by the prior

With center = 150, σ = 0.5:

| BPM | prior value |
|-----|-------------|
| 20  | ≈ 0.0003    |
| 60  | ≈ 0.19      |
| 120 | ≈ 0.90      |
| 300 | ≈ 0.38      |

A genuine 60 BPM track has to beat its 120 BPM comb competitor by ~5× on
raw comb score — achievable (the pair-product comb + bass-weighting helps)
but fragile. Below ~40 BPM and above ~250 BPM the prior effectively
suppresses the candidate: the range is searchable (buffer-eligibility
allows it once enough data has accumulated) but won't be picked unless the
comb score is dramatic.

**Fix — genre-aware prior center.** Accept a construction-time parameter
`prior_center_bpm` and expose presets: `Electronic(150)`, `Pop(120)`,
`Ballad(90)`. Single-line change in `BeatTracker::new`. Leaves current
behaviour as the default.

**Fix — two-pass estimation.** First pass with a wide/uniform prior picks
the dominant lag-family; second pass uses a narrow prior centered on the
first-pass estimate. Librosa does this. Cost: one extra comb sweep per
update (~0.5 s cadence, negligible).

### 3. Transient flux spikes cause BPM wobble

The onset ring buffer is normalised to its running max. A single loud
transient (a cymbal crash, a vocal shout) raises the floor and
temporarily lowers the relative magnitude of real beats until the
transient ages out of the normalisation window.

**Fix — local-median normalisation.** Replace `max` with `median + k·MAD`
(median-absolute-deviation) over the ring buffer. Robust to outliers;
~5 lines of code.

### 4. No confidence signal

Downstream code cannot tell "locked at 128 BPM" from "guessing between 85
and 170." Visualisations can't dim the BPM readout during unlock; tests
can't gate assertions on confidence.

**Fix — expose comb peak prominence.** Output
`confidence = (score(best) − score(second_best)) / score(best)`
alongside BPM. Cheap; peak-selection code already has both values.

### 5. Tempo lock is not sticky

Once the tracker is on 128 BPM for 30 s, a brief noisy section can still
pull it to 96 BPM (a 3/4 harmonic) within 3–4 update cycles. The 20%
per-update cap + EMA gives symmetric agility — it reacts the same whether
locked or exploring.

**Fix — hysteresis via lock-strength accumulator.**

```
if |new_bpm − bpm| < 2:  lock_strength = min(1.0, lock_strength + 0.05)
else:                    lock_strength = max(0.0, lock_strength − 0.20)

ema_weight      = 0.35 · (1 − 0.7·lock_strength)   // 0.35 → 0.105 when locked
change_cap_pct  = 0.20 · (1 − 0.7·lock_strength)   // 20% → 6%
```

Costs ~10 lines; stabilises committed tempos without preventing genuine
tempo changes (sustained disagreement erodes `lock_strength` within ~4 s).

### 6. Slow tempos get only one pair-set

The buffer is sized as `(N_PAIRS + 1) × max_lag`, so the slowest tempo
(`MIN_BPM = 20`) has its longest lag equal to `buffer_len / (N_PAIRS + 1)`
— exactly **one** pair-set, no averaging headroom. Fast tempos get
hundreds of averaging terms; the slowest get one. Tempos near `MIN_BPM`
are noisier as a result.

Note this is independent of the warmup decoupling: `WARMUP_MIN_FRAMES` only
controls *when* the first estimate fires, not how many pair-sets are
available at a given lag.

**Fix — multiply `history_len` by 2.** Changing the derivation to
`2 × (N_PAIRS + 1) × max_lag` doubles the buffer (2400 frames ≈ 24 s at
MIN_BPM = 20) and gives ~2× more averaging at long lags, with no effect on
fast-tempo warmup. Cost: more memory and a longer "full-range eligibility"
horizon for the slowest tempos.

### 7. Spectral flux misses pitched onsets

Flux detects energy jumps. A legato melodic phrase where notes change
pitch but total energy stays constant produces no onset — even if the
notes are aligned to beats.

**Fix — complex-domain onset function.** Adds phase-change detection to
the magnitude-change detection already present: when magnitude is
unchanged but phase jumps, a new note started. Roughly 2× cost of current
flux; robust for acoustic/melodic material. Not a priority for electronic
music (which Kaidee targets) but worth noting for future genre expansion.

### Out of scope for the DSP layer

These surface from the algorithm but belong to *test* or *evaluation*
code, not `beat.rs`:

- **Reference-data octave errors.** Per-segment librosa BPM sometimes
  reports 2× or ½× the global value (librosa has its own octave-error
  failure modes). The `reference_audio.rs` assertions should filter
  segments whose librosa BPM disagrees with the global estimate by more
  than a factor of 1.5, rather than expecting Kaidee to reproduce
  librosa's errors.
- **Ground-truth fixtures.** The ACIDSEX case reveals that *neither*
  librosa nor Kaidee is reliably correct on dense syncopated material.
  Hand-annotated beat timings for 2–3 hard fixtures would let the tests
  measure genuine accuracy rather than agreement with another tool.
