//! Beat tracker — onset-based BPM (beats per minute) estimation driving a
//! PLL (phase-locked loop) that keeps beat phase aligned with the music.
//! Full algorithm write-up: see `beat.md` next to this file.

/// Number of onset history frames kept for autocorrelation.
/// At ~100 Hz analysis rate this is ~8 seconds of history.
const ONSET_HISTORY: usize = 800;

pub struct BeatTracker {
    // --- Onset detection ---
    prev_magnitudes: Vec<f32>,
    n_bass_bins: usize, // how many bins fall below ~250 Hz (bass-weighted region)

    // --- Onset history ring buffer ---
    onset_buf: Vec<f32>,
    onset_write: usize,
    onset_count: usize,

    // --- BPM ---
    pub bpm: f32,
    bpm_timer: usize,

    // --- PLL ---
    pub phase: f32, // 0.0–1.0 position within current beat (0 = beat)

    analysis_rate: f32,
}

impl BeatTracker {
    pub fn new(sample_rate: f32, hop: usize, n_bins: usize) -> Self {
        let analysis_rate = sample_rate / hop as f32;
        let hz_per_bin = (sample_rate / 2.0) / n_bins as f32;
        // Bins below 250 Hz — kick and bass transients live here
        let n_bass_bins = (250.0 / hz_per_bin) as usize;

        Self {
            prev_magnitudes: vec![0.0; n_bins],
            n_bass_bins,
            onset_buf: vec![0.0; ONSET_HISTORY],
            onset_write: 0,
            onset_count: 0,
            bpm: 120.0,
            bpm_timer: 0,
            phase: 0.0,
            analysis_rate,
        }
    }

    /// Process one frame of FFT (fast Fourier transform) magnitudes.
    /// Returns (bpm, beat_phase, onset_strength).
    pub fn update(&mut self, magnitudes: &[f32]) -> (f32, f32, f32) {
        let onset = self.spectral_flux(magnitudes);

        self.onset_buf[self.onset_write] = onset;
        self.onset_write = (self.onset_write + 1) % ONSET_HISTORY;
        self.onset_count += 1;

        // Re-estimate BPM every ~1 second once we have a full history buffer
        if self.onset_count >= ONSET_HISTORY {
            if self.bpm_timer == 0 {
                self.bpm_timer = self.analysis_rate as usize;
                self.update_bpm();
            }
            self.bpm_timer = self.bpm_timer.saturating_sub(1);
        }

        // Advance PLL phase
        self.phase += self.bpm / 60.0 / self.analysis_rate;
        self.phase = self.phase.rem_euclid(1.0);

        // PLL correction on strong onsets — nudge phase toward nearest beat boundary.
        // Threshold raised to 0.3 so noise doesn't constantly pull the phase around.
        // Gain lowered to 0.12 for a gentler correction per frame.
        if onset > 0.3 {
            let err = if self.phase > 0.5 {
                self.phase - 1.0 // e.g. 0.9 → -0.1 (slightly before next beat)
            } else {
                self.phase // e.g. 0.1 → slightly after last beat
            };
            self.phase -= err * 0.12 * onset;
            self.phase = self.phase.rem_euclid(1.0);
        }

        (self.bpm, self.phase, onset)
    }

    /// Weighted log-magnitude spectral flux.
    ///
    /// Two tricks over plain flux:
    /// - **Log compression** `log(1 + C·mag)` maps raw magnitudes into a
    ///   perceptual scale, so quiet passages produce clean onsets and loud
    ///   passages don't saturate. Without this, the onset signal is dominated
    ///   by absolute loudness rather than relative energy arrivals.
    /// - **Bass weighting (3×)** for bins below 250 Hz — kick drum transients
    ///   are the dominant beat signal in electronic music; treating all bins
    ///   equally dilutes the kick with hi-hats, reverb tails, and noise.
    fn spectral_flux(&mut self, magnitudes: &[f32]) -> f32 {
        let len = magnitudes.len().min(self.prev_magnitudes.len());
        const COMPRESSION: f32 = 1000.0;

        let flux: f32 = magnitudes[..len]
            .iter()
            .zip(self.prev_magnitudes[..len].iter())
            .enumerate()
            .map(|(i, (&new, &old))| {
                let log_new = (1.0 + COMPRESSION * new).ln();
                let log_old = (1.0 + COMPRESSION * old).ln();
                let diff = (log_new - log_old).max(0.0);
                let weight = if i < self.n_bass_bins { 3.0 } else { 1.0 };
                diff * weight
            })
            .sum();

        self.prev_magnitudes[..len].copy_from_slice(&magnitudes[..len]);

        (flux / len as f32 * 2.0).clamp(0.0, 1.0)
    }

    fn update_bpm(&mut self) {
        let n = ONSET_HISTORY;
        let min_lag = (self.analysis_rate * 60.0 / 200.0) as usize; // 200 BPM
        let max_lag = (self.analysis_rate * 60.0 / 60.0) as usize; //  60 BPM

        // Number of harmonics in the comb filter. Scoring a lag as a pulse
        // train at L, 2L, 3L (instead of a single correlation at L) reinforces
        // genuine tempos across multiple periods and suppresses noise matches.
        const N_HARMONICS: usize = 3;

        // Log-Gaussian tempo prior: musically plausible tempos cluster around
        // 120 BPM. This breaks the octave-ambiguity tie between e.g. 70 and
        // 140 BPM, which score comparably on autocorrelation alone.
        const PRIOR_CENTER_BPM: f32 = 120.0;
        const PRIOR_SIGMA: f32 = 0.4; // in log-octaves
        let prior = |bpm: f32| -> f32 {
            let log_ratio = (bpm / PRIOR_CENTER_BPM).ln();
            (-log_ratio * log_ratio / (2.0 * PRIOR_SIGMA * PRIOR_SIGMA)).exp()
        };

        let mut best_lag = 0usize;
        let mut best_score = 0.0f32;

        // Need (N_HARMONICS+1)*lag ≤ n for at least one term in the sum.
        let lag_cap = max_lag.min(n / (N_HARMONICS + 1));

        for lag in min_lag..=lag_cap {
            let terms = n - N_HARMONICS * lag;

            // Comb filter: Σ_t onset[t] · (onset[t+L] + onset[t+2L] + onset[t+3L])
            // Normalized by the number of terms so the score isn't biased
            // toward shorter lags (which accumulate more terms).
            let corr: f32 = (0..terms)
                .map(|i| {
                    let t0 = (self.onset_write + i) % n;
                    let t1 = (self.onset_write + i + lag) % n;
                    let t2 = (self.onset_write + i + 2 * lag) % n;
                    let t3 = (self.onset_write + i + 3 * lag) % n;
                    self.onset_buf[t0]
                        * (self.onset_buf[t1] + self.onset_buf[t2] + self.onset_buf[t3])
                })
                .sum::<f32>()
                / terms as f32;

            let bpm = self.analysis_rate * 60.0 / lag as f32;
            let score = corr * prior(bpm);

            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }

        if best_lag == 0 {
            return;
        }

        let candidate = self.analysis_rate * 60.0 / best_lag as f32;
        if !(60.0..=200.0).contains(&candidate) {
            return;
        }

        // Cap how far BPM can move per update cycle — prevents a single noisy
        // frame from throwing the estimate across the range. 8% of current BPM
        // is ~10 BPM at 128 BPM, enough to track gradual tempo drift.
        let max_delta = self.bpm * 0.08;
        let clamped = candidate.clamp(self.bpm - max_delta, self.bpm + max_delta);

        // Slow exponential moving average — 15% weight on new estimate so
        // BPM converges steadily rather than jumping on each update.
        self.bpm = self.bpm * 0.85 + clamped * 0.15;
    }
}
