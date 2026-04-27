//! FFT (fast Fourier transform) spectral analyser — short-time FFT with a
//! Hann window.
//!
//! See `fft.md` next to this file for a full algorithm write-up.

use super::{BandEnergies, SpectralAnalyzer};
use super::super::levels::to_01;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Short-time FFT analyser with Hann windowing.
///
/// Maintains a circular sample buffer of `FFT_SIZE` samples. Each call to
/// [`process`] applies a Hann window, runs a forward FFT, and updates the
/// internal magnitude, band-energy, and onset state.
///
/// **Trade-offs vs other strategies:**
/// - Frequency resolution: ~21.5 Hz/bin at 44 100 Hz — coarse in the bass
///   range (sub-bass spans only 2 bins).
/// - Latency: one `FFT_SIZE`-sample window (~46 ms at 44 100 Hz).
/// - CPU: O(N log N) per frame — cheap at 100 Hz analysis rate.
/// - Spectral leakage: Hann window reduces sidelobes but cannot eliminate
///   them; non-integer-bin frequencies still spill into adjacent bands.
pub struct FftAnalyzer {
    fft: Arc<dyn Fft<f32>>,
    scratch: Vec<Complex<f32>>,
    hann: Vec<f32>,
    fft_buf: Vec<Complex<f32>>,
    magnitudes: Vec<f32>,
    prev_magnitudes: Vec<f32>,
    onset: f32,
    // Circular sample buffer — always holds the most recent FFT_SIZE samples.
    // New samples overwrite the oldest; write_pos is where the next write goes.
    window: Vec<f32>,
    write_pos: usize,
    sample_rate: f32,
}

impl FftAnalyzer {
    pub const FFT_SIZE: usize = 2048;

    pub fn new(sample_rate: f32) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FftAnalyzer::FFT_SIZE);
        let scratch = vec![Complex::new(0.0f32, 0.0); fft.get_inplace_scratch_len()];

        // Hann window — tapers the edges of each FFT frame to zero. Without
        // this, sharp edges at frame boundaries create false high-frequency
        // content.
        let hann: Vec<f32> = (0..FftAnalyzer::FFT_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (FftAnalyzer::FFT_SIZE as f32 - 1.0))
                        .cos())
            })
            .collect();

        let n_bins = FftAnalyzer::FFT_SIZE / 2;
        Self {
            fft,
            scratch,
            hann,
            fft_buf: vec![Complex::new(0.0f32, 0.0); FftAnalyzer::FFT_SIZE],
            magnitudes: vec![0.0f32; n_bins],
            prev_magnitudes: vec![0.0f32; n_bins],
            onset: 0.0,
            window: vec![0.0f32; FftAnalyzer::FFT_SIZE],
            write_pos: 0,
            sample_rate,
        }
    }

    fn band_rms(&self, lo_hz: f32, hi_hz: f32) -> f32 {
        let hz_per_bin = (self.sample_rate / 2.0) / self.magnitudes.len() as f32;
        let lo = (lo_hz / hz_per_bin) as usize;
        let hi = ((hi_hz / hz_per_bin) as usize + 1).min(self.magnitudes.len());
        if lo >= hi {
            return 0.0;
        }
        // Total RMS (root mean square) energy within a frequency band
        // [lo_hz, hi_hz). Sum of squared magnitudes (not mean) so wide bands
        // (MID = 71 bins, AIR = 744 bins) get proportional credit relative to
        // narrow bands (SUB = 2 bins, BASS = 9 bins).
        let total_sq: f32 = self.magnitudes[lo..hi].iter().map(|m| m * m).sum();
        total_sq.sqrt()
    }

    /// Weighted log-magnitude spectral flux.
    ///
    /// Two tricks over plain flux:
    /// - **Log compression** `log(1 + C·mag)` maps raw magnitudes into a
    ///   perceptual scale, so quiet passages produce clean onsets and loud
    ///   passages don't saturate.
    /// - **Bass weighting (3×)** for bins below 250 Hz — kick drum transients
    ///   are the dominant beat signal in electronic music; treating all bins
    ///   equally dilutes the kick with hi-hats, reverb tails, and noise.
    fn compute_onset(&self) -> f32 {
        let len = self.magnitudes.len();
        let hz_per_bin = (self.sample_rate / 2.0) / len as f32;
        let n_bass_bins = (250.0 / hz_per_bin) as usize;
        const COMPRESSION: f32 = 1000.0;

        let flux: f32 = self.magnitudes[..len]
            .iter()
            .zip(self.prev_magnitudes[..len].iter())
            .enumerate()
            .map(|(i, (&new, &old))| {
                let log_new = (1.0 + COMPRESSION * new).ln();
                let log_old = (1.0 + COMPRESSION * old).ln();
                let diff = (log_new - log_old).max(0.0);
                let weight = if i < n_bass_bins { 3.0 } else { 1.0 };
                diff * weight
            })
            .sum();

        (flux / len as f32 * 2.0).clamp(0.0, 1.0)
    }
}

impl SpectralAnalyzer for FftAnalyzer {
    fn push_sample(&mut self, sample: f32) {
        self.window[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % FftAnalyzer::FFT_SIZE;
    }

    fn process(&mut self) {
        for i in 0..FftAnalyzer::FFT_SIZE {
            let idx = (self.write_pos + i) % FftAnalyzer::FFT_SIZE;
            self.fft_buf[i] = Complex::new(self.window[idx] * self.hann[i], 0.0);
        }

        self.fft
            .process_with_scratch(&mut self.fft_buf, &mut self.scratch);

        // Only the first half of FFT output is useful (positive frequencies).
        // The second half mirrors it (Nyquist symmetry for real input).
        // Normalize by FFT_SIZE so magnitudes don't scale with window size.
        let n_bins = FftAnalyzer::FFT_SIZE / 2;
        for (i, c) in self.fft_buf[..n_bins].iter().enumerate() {
            self.magnitudes[i] = c.norm() * 2.0 / FftAnalyzer::FFT_SIZE as f32;
        }

        self.onset = self.compute_onset();
        self.prev_magnitudes.copy_from_slice(&self.magnitudes);
    }

    fn magnitudes(&self) -> &[f32] {
        &self.magnitudes
    }

    fn band_energies(&self) -> BandEnergies {
        BandEnergies {
            sub_bass: to_01(self.band_rms(20.0, 60.0)),
            bass: to_01(self.band_rms(60.0, 250.0)),
            low_mid: to_01(self.band_rms(250.0, 500.0)),
            mid: to_01(self.band_rms(500.0, 2000.0)),
            presence: to_01(self.band_rms(2000.0, 6000.0)),
            air: to_01(self.band_rms(6000.0, self.sample_rate / 2.0)),
        }
    }

    fn onset_strength(&self) -> f32 {
        self.onset
    }

    fn n_bins(&self) -> usize {
        FftAnalyzer::FFT_SIZE / 2
    }
}

#[cfg(test)]
mod tests {
    use super::FftAnalyzer;
    use super::super::{BandEnergies, SpectralAnalyzer};

    const SR: f32 = 44100.0;

    fn analyze(samples: &[f32]) -> BandEnergies {
        let mut s = FftAnalyzer::new(SR);
        for &x in samples {
            s.push_sample(x);
        }
        s.process();
        s.band_energies()
    }

    fn sine(freq_hz: f32, duration_secs: f32) -> Vec<f32> {
        let n = (SR * duration_secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / SR).sin())
            .collect()
    }

    #[test]
    fn silence_all_bands_zero() {
        // All-zero input → zero magnitudes → to_01 clamps below −80 dB to 0.0 exactly.
        let b = analyze(&vec![0.0f32; 44100]);
        assert_eq!(b.sub_bass, 0.0);
        assert_eq!(b.bass, 0.0);
        assert_eq!(b.mid, 0.0);
        assert_eq!(b.air, 0.0);
    }

    #[test]
    fn sine_40hz_activates_sub_bass_and_bass() {
        // 40 Hz falls between FFT bins (~bin 1.86 at 44 100 Hz), so the Hann
        // main lobe straddles the sub-bass/bass boundary. Both bands light up.
        // All higher bands are far enough away to receive no energy.
        let b = analyze(&sine(40.0, 1.0));
        assert!(b.sub_bass > 0.9, "sub_bass={:.3}", b.sub_bass);
        assert!(b.bass > 0.9, "bass={:.3}", b.bass);
        assert!(b.low_mid < 0.1, "low_mid={:.3}", b.low_mid);
        assert_eq!(b.mid, 0.0);
        assert_eq!(b.presence, 0.0);
        assert_eq!(b.air, 0.0);
    }

    #[test]
    fn sine_150hz_activates_bass() {
        // 150 Hz is squarely in bass (60–250 Hz). Sub-bass gets a small spill
        // (~0.036) and low-mid gets adjacent leakage (~0.12). Mid and above
        // are exact zero.
        let b = analyze(&sine(150.0, 1.0));
        assert!(b.bass > 0.9, "bass={:.3}", b.bass);
        assert!(b.sub_bass < 0.05, "sub_bass={:.3}", b.sub_bass);
        assert!(b.low_mid < 0.15, "low_mid={:.3}", b.low_mid);
        assert_eq!(b.mid, 0.0);
        assert_eq!(b.presence, 0.0);
        assert_eq!(b.air, 0.0);
    }

    #[test]
    fn sine_350hz_activates_low_mid() {
        // 350 Hz is inside low-mid (250–500 Hz). Bass and mid are adjacent
        // and both pick up some leakage; sub-bass, presence, and air are exact
        // zero.
        let b = analyze(&sine(350.0, 1.0));
        assert!(b.low_mid > 0.9, "low_mid={:.3}", b.low_mid);
        assert_eq!(b.sub_bass, 0.0);
        assert!(b.bass < 0.3, "bass={:.3}", b.bass);
        assert!(b.mid < 0.2, "mid={:.3}", b.mid);
        assert_eq!(b.presence, 0.0);
        assert_eq!(b.air, 0.0);
    }

    #[test]
    fn sine_1khz_isolated_in_mid() {
        // 1 000 Hz is squarely in mid (500–2 000 Hz). Sub-bass, bass, low-mid,
        // and presence are all exact zero — they're either narrow or far enough
        // away that no energy reaches them. Air spans 744 bins so residual
        // leakage accumulates to ~0.03; use a threshold there.
        let b = analyze(&sine(1000.0, 1.0));
        assert!(b.mid > 0.9, "mid={:.3}", b.mid);
        assert_eq!(b.sub_bass, 0.0);
        assert_eq!(b.bass, 0.0);
        assert_eq!(b.low_mid, 0.0);
        assert_eq!(b.presence, 0.0);
        assert!(b.air < 0.05, "air={:.3}", b.air);
    }

    #[test]
    fn sine_4khz_activates_presence() {
        // 4 000 Hz is inside presence (2 000–6 000 Hz). Air is adjacent (186
        // bins) and accumulates ~0.19. All other bands are either exact zero
        // or negligibly small.
        let b = analyze(&sine(4000.0, 1.0));
        assert!(b.presence > 0.9, "presence={:.3}", b.presence);
        assert_eq!(b.bass, 0.0);
        assert_eq!(b.low_mid, 0.0);
        assert!(b.sub_bass < 0.05, "sub_bass={:.3}", b.sub_bass);
        assert!(b.mid < 0.05, "mid={:.3}", b.mid);
        assert!(b.air < 0.2, "air={:.3}", b.air);
    }

    #[test]
    fn sine_8khz_isolated_in_air() {
        // 8 000 Hz is inside air (6 000 Hz–Nyquist). Sub-bass, bass, and
        // low-mid are exact zero. Mid (71 bins) accumulates residuals to ~0.07
        // and presence (186 bins, adjacent) to ~0.17 — both need thresholds.
        let b = analyze(&sine(8000.0, 1.0));
        assert!(b.air > 0.9, "air={:.3}", b.air);
        assert_eq!(b.sub_bass, 0.0);
        assert_eq!(b.bass, 0.0);
        assert_eq!(b.low_mid, 0.0);
        assert!(b.mid < 0.1, "mid={:.3}", b.mid);
        assert!(b.presence < 0.2, "presence={:.3}", b.presence);
    }

    #[test]
    fn onset_fires_on_attack() {
        // Silence followed by a loud bass burst. The first frame after the
        // burst should register a strong onset.
        let mut s = FftAnalyzer::new(SR);
        // Prime with silence so prev_magnitudes are near zero.
        for _ in 0..FftAnalyzer::FFT_SIZE {
            s.push_sample(0.0);
        }
        s.process();
        assert_eq!(s.onset_strength(), 0.0, "silence should produce zero onset");

        // Now feed a full-scale 100 Hz sine.
        for i in 0..FftAnalyzer::FFT_SIZE {
            let t = i as f32 / SR;
            s.push_sample((2.0 * std::f32::consts::PI * 100.0 * t).sin());
        }
        s.process();
        assert!(
            s.onset_strength() > 0.1,
            "onset={:.3} expected > 0.1 after bass attack",
            s.onset_strength()
        );
    }

    #[test]
    fn onset_decays_on_steady_state() {
        // After several frames of a sustained sine, prev_magnitudes converge
        // to magnitudes and the positive flux drops toward zero.
        let mut s = FftAnalyzer::new(SR);
        let mut onset = 1.0f32;
        for frame in 0..20 {
            for i in 0..FftAnalyzer::FFT_SIZE {
                let t = (frame * FftAnalyzer::FFT_SIZE + i) as f32 / SR;
                s.push_sample((2.0 * std::f32::consts::PI * 100.0 * t).sin());
            }
            s.process();
            onset = s.onset_strength();
        }
        assert!(onset < 0.05, "onset={:.3} expected < 0.05 on steady sine", onset);
    }
}
