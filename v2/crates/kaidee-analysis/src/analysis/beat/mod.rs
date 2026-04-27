//! Beat tracker — onset-based BPM (beats per minute) estimation driving a
//! PLL (phase-locked loop) that keeps beat phase aligned with the music.
//! Full algorithm write-up: see `beat.md` next to this file.
//!
//! The tracker consumes a scalar onset strength per frame from the spectral
//! analyzer. It has no knowledge of frequency bins or magnitudes — onset
//! computation is the spectral layer's responsibility. Swapping the spectral
//! strategy will change the onset signal's characteristics and may require
//! re-tuning the constants below.

use crate::global::ANALYSIS_RATE_HZ;

/// Fixed operational BPM range. PRIOR_CENTER_BPM (150) is tuned around
/// typical dance tempos; the extremes rely on buffer-size eligibility and
/// the fast-harmonic probe rather than prior support.
const MIN_BPM: f32 = 20.0;
const MAX_BPM: f32 = 300.0;

/// Pair-product comb order — pairs per lag.
const N_PAIRS: usize = 3;

/// How often BPM is re-estimated.
const BPM_UPDATE_INTERVAL_SECS: f32 = 0.5;
const BPM_UPDATE_INTERVAL_FRAMES: usize = (BPM_UPDATE_INTERVAL_SECS * ANALYSIS_RATE_HZ) as usize;

/// Fastest tempo guaranteed eligible at the first BPM estimate. Sets the
/// warmup length via the dynamic lag cap: `lag_cap = available / (N_PAIRS+1)`
/// must be ≥ the lag of WARMUP_MIN_BPM.
const WARMUP_MIN_BPM: f32 = 60.0;
const WARMUP_MIN_FRAMES: usize =
    ((N_PAIRS + 1) as f32 * ANALYSIS_RATE_HZ * 60.0 / WARMUP_MIN_BPM) as usize;

/// Log-Gaussian tempo prior. σ is in natural-log units of the BPM ratio.
const PRIOR_CENTER_BPM: f32 = 150.0;
const PRIOR_SIGMA: f32 = 0.5;

/// PLL phase-correction gain and threshold. Gain is kept small so the PLL
/// tracks without snapping; threshold gates out low-confidence onsets.
const PLL_GAIN: f32 = 0.12;
const PLL_THRESHOLD: f32 = 0.3;

/// Per-update BPM shift cap (fraction of current BPM) and EMA weight on
/// the new estimate.
const BPM_CHANGE_CAP: f32 = 0.20;
const BPM_EMA_WEIGHT: f32 = 0.35;

/// Onset ring buffer length: one pair-set at MIN_BPM.
fn history_len() -> usize {
    let max_lag = (ANALYSIS_RATE_HZ * 60.0 / MIN_BPM) as usize;
    (N_PAIRS + 1) * max_lag
}

pub struct BeatTracker {
    // --- Onset history ring buffer ---
    onset_buf: Vec<f32>,
    onset_write: usize,
    onset_count: usize,

    // --- BPM ---
    pub bpm: f32,

    // --- PLL ---
    pub phase: f32, // 0.0–1.0 position within current beat (0 = beat)
}

impl BeatTracker {
    pub fn new() -> Self {
        Self {
            onset_buf: vec![0.0; history_len()],
            onset_write: 0,
            onset_count: 0,
            bpm: 120.0,
            phase: 0.0,
        }
    }

    /// Process one onset strength value from the spectral analyzer.
    /// Returns (bpm, beat_phase, onset_strength).
    pub fn update(&mut self, onset: f32) -> (f32, f32, f32) {
        let n = self.onset_buf.len();
        self.onset_buf[self.onset_write] = onset;
        self.onset_write = (self.onset_write + 1) % n;
        self.onset_count += 1;

        // Short-circuit on `>= WARMUP_MIN_FRAMES` keeps the subtraction from
        // underflowing during warmup.
        if self.onset_count >= WARMUP_MIN_FRAMES
            && (self.onset_count - WARMUP_MIN_FRAMES) % BPM_UPDATE_INTERVAL_FRAMES == 0
        {
            self.update_bpm();
        }

        self.pll_step(onset);

        (self.bpm, self.phase, onset)
    }

    fn pll_step(&mut self, onset: f32) {
        self.phase += self.bpm / 60.0 / ANALYSIS_RATE_HZ;
        self.phase = self.phase.rem_euclid(1.0);

        if onset > PLL_THRESHOLD {
            let err = if self.phase > 0.5 {
                self.phase - 1.0
            } else {
                self.phase
            };
            self.phase -= err * PLL_GAIN * onset;
            self.phase = self.phase.rem_euclid(1.0);
        }
    }

    fn update_bpm(&mut self) {
        let n = self.onset_buf.len();
        let min_lag = (ANALYSIS_RATE_HZ * 60.0 / MAX_BPM) as usize;
        let max_lag = (ANALYSIS_RATE_HZ * 60.0 / MIN_BPM) as usize;

        // Valid-data window: pre-wrap it's [0, onset_count); post-wrap it's
        // the full ring starting at onset_write (the oldest slot, about to be
        // overwritten next).
        let (start, available) = if self.onset_count < n {
            (0, self.onset_count)
        } else {
            (self.onset_write, n)
        };

        let prior = |bpm: f32| -> f32 {
            let log_ratio = (bpm / PRIOR_CENTER_BPM).ln();
            (-log_ratio * log_ratio / (2.0 * PRIOR_SIGMA * PRIOR_SIGMA)).exp()
        };

        let comb_corr = |lag: usize| -> f32 {
            let terms = available - N_PAIRS * lag;
            (0..terms)
                .map(|i| {
                    let t0 = (start + i) % n;
                    let t1 = (start + i + lag) % n;
                    let t2 = (start + i + 2 * lag) % n;
                    let t3 = (start + i + 3 * lag) % n;
                    self.onset_buf[t0] * self.onset_buf[t1]
                        + self.onset_buf[t1] * self.onset_buf[t2]
                        + self.onset_buf[t2] * self.onset_buf[t3]
                })
                .sum::<f32>()
                / terms as f32
        };
        let comb_score = |lag: usize| -> f32 {
            comb_corr(lag) * prior(ANALYSIS_RATE_HZ * 60.0 / lag as f32)
        };

        // Only test tempos whose comb window fits in the data collected so
        // far. Slow tempos become eligible as `available` grows.
        let lag_cap = max_lag.min(available / (N_PAIRS + 1));

        let mut best_lag: usize = 0;
        let mut best_score: f32 = 0.0;
        for lag in min_lag..=lag_cap {
            let score = comb_score(lag);
            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }

        if best_lag == 0 {
            return;
        }

        // Fast-harmonic probe — see beat.md.
        const FAST_HARMONIC_THRESHOLD: f32 = 0.95;
        let mut best_corr = comb_corr(best_lag);
        while best_lag / 2 >= min_lag {
            let half = best_lag / 2;
            let half_corr = comb_corr(half);
            if half_corr >= FAST_HARMONIC_THRESHOLD * best_corr {
                best_lag = half;
                best_corr = half_corr;
            } else {
                break;
            }
        }
        best_score = comb_score(best_lag);

        let lag_f = if best_lag > min_lag && best_lag < lag_cap {
            let s_prev = comb_score(best_lag - 1);
            let s_curr = best_score;
            let s_next = comb_score(best_lag + 1);
            let denom = s_prev - 2.0 * s_curr + s_next;
            if denom < 0.0 {
                // Parabola opens downward: valid peak
                let delta = 0.5 * (s_prev - s_next) / denom;
                best_lag as f32 + delta.clamp(-1.0, 1.0)
            } else {
                best_lag as f32
            }
        } else {
            best_lag as f32
        };

        let candidate = ANALYSIS_RATE_HZ * 60.0 / lag_f;
        if !(MIN_BPM..=MAX_BPM).contains(&candidate) {
            return;
        }

        let max_delta = self.bpm * BPM_CHANGE_CAP;
        let clamped = candidate.clamp(self.bpm - max_delta, self.bpm + max_delta);
        self.bpm = self.bpm * (1.0 - BPM_EMA_WEIGHT) + clamped * BPM_EMA_WEIGHT;
    }
}

#[cfg(test)]
mod tests {
    use super::super::pipeline::Pipeline;

    const SR: f32 = 44100.0;
    const TOLERANCE: f32 = 0.5;

    /// Short-lag tempos have coarser parabolic-interpolation resolution: a
    /// single-frame integer lag is ~10× wider in BPM at lag 20 than at lag
    /// 200, so sub-frame onset spread from the FFT shifts the interpolated
    /// peak by ~0.5–1 BPM. Tolerate ±1 BPM at the high end.
    const HIGH_BPM_TOLERANCE: f32 = 1.0;

    /// Build a click track at `target_bpm` and run the pipeline for `duration_secs`.
    /// Returns the BPM from the last produced frame.
    fn run_click_track(target_bpm: f32, duration_secs: f32) -> f32 {
        let beat_samples = (SR * 60.0 / target_bpm) as usize;
        let total = (SR * duration_secs) as usize;
        let mut samples = vec![0.0f32; total];
        for i in (0..total).step_by(beat_samples) {
            samples[i] = 1.0;
        }
        let mut pipeline = Pipeline::new(SR, 1);
        let frames = pipeline.push(&samples);
        frames.last().expect("no frames produced").bpm
    }

    #[test]
    fn click_track_20bpm_converges() {
        // ~12 s buffer-fill to first estimate + ~12 s clamped descent from
        // 120 BPM + ~2 s EMA settling → 28 s.
        let bpm = run_click_track(20.0, 28.0);
        assert!(
            (20.0 - TOLERANCE..=20.0 + TOLERANCE).contains(&bpm),
            "20 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            20.0
        );
    }

    #[test]
    fn click_track_30bpm_converges() {
        // ~8 s to eligibility + ~8 s clamped descent + ~2 s settling → 20 s.
        let bpm = run_click_track(30.0, 20.0);
        assert!(
            (30.0 - TOLERANCE..=30.0 + TOLERANCE).contains(&bpm),
            "30 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            30.0
        );
    }

    #[test]
    fn click_track_40bpm_converges() {
        // ~6 s to eligibility + ~6 s clamped descent + ~2 s settling → 16 s.
        let bpm = run_click_track(40.0, 16.0);
        assert!(
            (40.0 - TOLERANCE..=40.0 + TOLERANCE).contains(&bpm),
            "40 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            40.0
        );
    }

    #[test]
    fn click_track_50bpm_converges() {
        // ~5 s to eligibility + ~4.5 s clamped descent + ~2 s settling → 14 s.
        let bpm = run_click_track(50.0, 14.0);
        assert!(
            (50.0 - TOLERANCE..=50.0 + TOLERANCE).contains(&bpm),
            "50 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            50.0
        );
    }

    #[test]
    fn click_track_70bpm_converges() {
        // 4 s warmup + ~7 s descent from 120 BPM + ~3 s EMA settling → 18 s.
        let bpm = run_click_track(70.0, 18.0);
        assert!(
            (70.0 - TOLERANCE..=70.0 + TOLERANCE).contains(&bpm),
            "70 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            70.0
        );
    }

    #[test]
    fn click_track_90bpm_converges() {
        // 4 s warmup + ~2 s climb + ~3 s EMA settling → 14 s.
        let bpm = run_click_track(90.0, 14.0);
        assert!(
            (90.0 - TOLERANCE..=90.0 + TOLERANCE).contains(&bpm),
            "90 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            90.0
        );
    }

    #[test]
    fn click_track_120bpm_converges() {
        // 4 s warmup + ~3 s EMA settling → 10 s.
        let bpm = run_click_track(120.0, 10.0);
        assert!(
            (120.0 - TOLERANCE..=120.0 + TOLERANCE).contains(&bpm),
            "120 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            120.0
        );
    }

    #[test]
    fn click_track_140bpm_converges() {
        // 4 s warmup + ~2.5 s climb + ~3 s EMA settling → 14 s.
        let bpm = run_click_track(140.0, 14.0);
        assert!(
            (140.0 - TOLERANCE..=140.0 + TOLERANCE).contains(&bpm),
            "140 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            140.0
        );
    }

    #[test]
    fn click_track_175bpm_converges() {
        // 4 s warmup + ~4 s climb + ~3 s EMA settling → 16 s.
        let bpm = run_click_track(175.0, 16.0);
        assert!(
            (175.0 - TOLERANCE..=175.0 + TOLERANCE).contains(&bpm),
            "175 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            175.0
        );
    }

    #[test]
    fn click_track_200bpm_converges() {
        // 4 s warmup + ~5.5 s climb + ~3 s EMA settling → 18 s.
        let bpm = run_click_track(200.0, 18.0);
        assert!(
            (200.0 - TOLERANCE..=200.0 + TOLERANCE).contains(&bpm),
            "200 BPM: got {:.1}, expected {:.1} ± {TOLERANCE}",
            bpm,
            200.0
        );
    }

    /// Exercises the fast-harmonic probe. Without it, the prior locks the
    /// tracker at 125 BPM (half-octave). With the probe, `best_lag` halves
    /// from 48 → 24 and the tracker reports the true 250 BPM.
    #[test]
    fn click_track_250bpm_converges() {
        // 4 s warmup + ~6 s climb + ~3 s EMA settling → 14 s.
        let bpm = run_click_track(250.0, 14.0);
        assert!(
            (250.0 - HIGH_BPM_TOLERANCE..=250.0 + HIGH_BPM_TOLERANCE).contains(&bpm),
            "250 BPM: got {:.1}, expected {:.1} ± {HIGH_BPM_TOLERANCE}",
            bpm,
            250.0
        );
    }

    #[test]
    fn click_track_300bpm_converges() {
        // 4 s warmup + ~7 s climb + ~3 s EMA settling → 16 s.
        let bpm = run_click_track(300.0, 16.0);
        assert!(
            (300.0 - HIGH_BPM_TOLERANCE..=300.0 + HIGH_BPM_TOLERANCE).contains(&bpm),
            "300 BPM: got {:.1}, expected {:.1} ± {HIGH_BPM_TOLERANCE}",
            bpm,
            300.0
        );
    }
}
