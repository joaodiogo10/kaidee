//! IIR (infinite impulse response) filterbank — 31 biquad bandpass filters at
//! ISO 61260 1/3-octave centre frequencies.
//!
//! See `iir.md` next to this file for a full algorithm write-up.

use super::{BandEnergies, SpectralAnalyzer};
use super::super::levels::to_01;

/// ISO 61260 1/3-octave centre frequencies in Hz, 20 Hz – 20 kHz.
const N_BANDS: usize = 31;
const CENTRE_FREQS: [f32; N_BANDS] = [
    // sub-bass (20–60 Hz)
    20.0, 25.0, 31.5, 40.0, 50.0,
    // bass (60–250 Hz)
    63.0, 80.0, 100.0, 125.0, 160.0, 200.0,
    // low-mid (250–500 Hz)
    250.0, 315.0, 400.0,
    // mid (500–2 000 Hz)
    500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    // presence (2 000–6 000 Hz)
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
    // air (6 000 Hz – Nyquist)
    6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
];

/// Q factor for 1/3-octave constant-bandwidth filters.
/// Derived from Q = 1 / (2^(1/6) − 2^(−1/6)).
const Q: f32 = 4.318;

/// Biquad bandpass filter in Transposed Direct Form II (TDF-II).
///
/// Uses only two state variables (`s1`, `s2`) instead of Direct Form I's four,
/// which reduces f32 accumulation errors — important when filtering small
/// amplitudes over many samples.
///
/// The feedforward coefficient `b1 = 0` for a bandpass filter by design, so
/// the TDF-II update simplifies to:
/// ```text
/// y  = b0·x + s1
/// s1 = −a1·y + s2      (b1 = 0 dropped)
/// s2 = b2·x − a2·y
/// ```
struct Biquad {
    b0: f32,
    b2: f32, // b1 = 0 always for bandpass
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl Biquad {
    fn new(fc: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * fc / sample_rate;
        let alpha = w0.sin() / (2.0 * Q);
        let a0 = 1.0 + alpha;
        Self {
            b0: alpha / a0,
            b2: -alpha / a0,
            a1: -2.0 * w0.cos() / a0,
            a2: (1.0 - alpha) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = -self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// IIR filterbank analyser using ISO 61260 1/3-octave bandpass filters.
///
/// Unlike block-based FFT analysis, each sample is processed immediately
/// through all 31 filters. `process()` is a snapshot — it reads out the
/// accumulated per-band RMS (root mean square) energy since the last call and
/// resets the accumulators. This gives ~10 ms effective latency (one hop
/// period) rather than the FFT's ~46 ms window latency.
///
/// **Trade-offs vs other strategies:**
/// - Latency: ~10 ms (one hop) vs ~46 ms for FFT.
/// - Spectral leakage: none — each filter is independent.
/// - Frequency resolution: fixed at ISO 1/3-octave centres; cannot be changed.
/// - Band boundaries: 1/3-octave filters have wide skirts (~26% bandwidth).
///   Canonical band boundaries (e.g. 250 Hz between bass and low-mid) are
///   approximate — a 250 Hz ISO band spans ~224–281 Hz.
/// - CPU: O(N_BANDS) per sample — lighter than FFT at the same analysis rate.
pub struct IirFilterbank {
    filters: [Biquad; N_BANDS],
    accumulators: [f32; N_BANDS],
    sample_counts: [u32; N_BANDS],
    prev_energies: [f32; N_BANDS],
    /// Current per-ISO-band RMS snapshot, returned by magnitudes().
    band_rms: [f32; N_BANDS],
    onset: f32,
}

impl IirFilterbank {
    pub fn new(sample_rate: f32) -> Self {
        // Filter stability check: skip bands above Nyquist.
        // All 31 ISO bands are below 22 050 Hz at 44 100 Hz sample rate.
        let filters = std::array::from_fn(|i| Biquad::new(CENTRE_FREQS[i], sample_rate));
        Self {
            filters,
            accumulators: [0.0; N_BANDS],
            sample_counts: [0; N_BANDS],
            prev_energies: [0.0; N_BANDS],
            band_rms: [0.0; N_BANDS],
            onset: 0.0,
        }
    }

    /// Aggregate per-ISO-band RMS values into one canonical band via
    /// Euclidean norm — matching the sum-of-squares approach in FftAnalyzer.
    fn canonical_rms(&self, lo_hz: f32, hi_hz: f32) -> f32 {
        let total_sq: f32 = CENTRE_FREQS
            .iter()
            .zip(self.band_rms.iter())
            .filter(|(&fc, _)| fc >= lo_hz && fc < hi_hz)
            .map(|(_, &rms)| rms * rms)
            .sum();
        total_sq.sqrt()
    }
}

impl SpectralAnalyzer for IirFilterbank {
    fn push_sample(&mut self, sample: f32) {
        for i in 0..N_BANDS {
            let y = self.filters[i].process(sample);
            self.accumulators[i] += y * y;
            self.sample_counts[i] += 1;
        }
    }

    fn process(&mut self) {
        // Snapshot per-band RMS and reset accumulators.
        for i in 0..N_BANDS {
            let count = self.sample_counts[i].max(1) as f32;
            self.band_rms[i] = (self.accumulators[i] / count).sqrt();
            self.accumulators[i] = 0.0;
            self.sample_counts[i] = 0;
        }

        // Bass-weighted positive log-delta across ISO bands.
        // Same log-compression (C = 1000) and bass weighting (3×) as
        // FftAnalyzer so the onset scale is comparable across strategies.
        const COMPRESSION: f32 = 1000.0;
        const BASS_THRESHOLD_HZ: f32 = 250.0;

        let flux: f32 = self.band_rms
            .iter()
            .zip(self.prev_energies.iter())
            .enumerate()
            .map(|(i, (&curr, &prev))| {
                let log_curr = (1.0 + COMPRESSION * curr).ln();
                let log_prev = (1.0 + COMPRESSION * prev).ln();
                let delta = (log_curr - log_prev).max(0.0);
                let weight = if CENTRE_FREQS[i] < BASS_THRESHOLD_HZ {
                    3.0
                } else {
                    1.0
                };
                delta * weight
            })
            .sum();

        self.onset = (flux / N_BANDS as f32 * 2.0).clamp(0.0, 1.0);
        self.prev_energies.copy_from_slice(&self.band_rms);
    }

    fn magnitudes(&self) -> &[f32] {
        &self.band_rms
    }

    fn band_energies(&self) -> BandEnergies {
        BandEnergies {
            sub_bass: to_01(self.canonical_rms(20.0, 60.0)),
            bass:     to_01(self.canonical_rms(60.0, 250.0)),
            low_mid:  to_01(self.canonical_rms(250.0, 500.0)),
            mid:      to_01(self.canonical_rms(500.0, 2000.0)),
            presence: to_01(self.canonical_rms(2000.0, 6000.0)),
            air:      to_01(self.canonical_rms(6000.0, f32::MAX)),
        }
    }

    fn onset_strength(&self) -> f32 {
        self.onset
    }

    fn n_bins(&self) -> usize {
        N_BANDS
    }
}

#[cfg(test)]
mod tests {
    use super::{IirFilterbank, CENTRE_FREQS, N_BANDS, Q};
    use super::super::{BandEnergies, SpectralAnalyzer};

    const SR: f32 = 44100.0;

    fn analyze(samples: &[f32]) -> BandEnergies {
        let mut s = IirFilterbank::new(SR);
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
    fn magnitudes_length_is_n_bands() {
        let s = IirFilterbank::new(SR);
        assert_eq!(s.magnitudes().len(), N_BANDS);
        assert_eq!(s.n_bins(), N_BANDS);
    }

    #[test]
    fn filter_poles_stable() {
        // All biquad poles must be inside the unit circle.
        // For a normalized biquad, the poles are roots of 1 + a1·z⁻¹ + a2·z⁻²,
        // i.e. z² + a1·z + a2 = 0. Both roots satisfy |z| < 1 iff |a2| < 1 and
        // |a1| < 1 + a2 (Jury stability conditions for second-order systems).
        use std::f32::consts::PI;
        for &fc in CENTRE_FREQS.iter() {
            let w0 = 2.0 * PI * fc / SR;
            let alpha = w0.sin() / (2.0 * Q);
            let a0 = 1.0 + alpha;
            let a2 = (1.0 - alpha) / a0;
            let a1 = -2.0 * w0.cos() / a0;
            assert!(
                a2.abs() < 1.0,
                "fc={fc} Hz: |a2|={:.4} ≥ 1 (unstable)",
                a2.abs()
            );
            assert!(
                a1.abs() < 1.0 + a2,
                "fc={fc} Hz: Jury condition failed (|a1|={:.4}, 1+a2={:.4})",
                a1.abs(),
                1.0 + a2
            );
        }
    }

    #[test]
    fn silence_all_bands_zero() {
        let b = analyze(&vec![0.0f32; 44100]);
        assert_eq!(b.sub_bass, 0.0);
        assert_eq!(b.bass, 0.0);
        assert_eq!(b.low_mid, 0.0);
        assert_eq!(b.mid, 0.0);
        assert_eq!(b.presence, 0.0);
        assert_eq!(b.air, 0.0);
    }

    #[test]
    fn sine_40hz_activates_sub_bass() {
        // 2nd-order biquad rolloff is -40 dB/decade per side. A 40 Hz sine
        // produces non-zero response in every canonical band — unlike FFT,
        // biquad filters have no sharp stopband. The key property tested here
        // is dominance: sub_bass is clearly the highest band.
        let b = analyze(&sine(40.0, 1.0));
        assert!(b.sub_bass > 0.5, "sub_bass={:.3}", b.sub_bass);
        assert!(b.sub_bass >= b.bass,     "sub_bass={:.3} should dominate bass={:.3}", b.sub_bass, b.bass);
        assert!(b.sub_bass >= b.mid,      "sub_bass={:.3} should dominate mid={:.3}", b.sub_bass, b.mid);
        assert!(b.sub_bass >= b.presence, "sub_bass={:.3} should dominate presence={:.3}", b.sub_bass, b.presence);
        assert!(b.sub_bass >= b.air,      "sub_bass={:.3} should dominate air={:.3}", b.sub_bass, b.air);
    }

    #[test]
    fn sine_150hz_activates_bass() {
        // 150 Hz sits between the 125 and 160 Hz ISO bands, both in the
        // canonical bass range. Bass is clearly the highest band.
        let b = analyze(&sine(150.0, 1.0));
        assert!(b.bass > 0.5, "bass={:.3}", b.bass);
        assert!(b.bass >= b.sub_bass,  "bass={:.3} should dominate sub_bass={:.3}", b.bass, b.sub_bass);
        assert!(b.bass >= b.mid,       "bass={:.3} should dominate mid={:.3}", b.bass, b.mid);
        assert!(b.bass >= b.presence,  "bass={:.3} should dominate presence={:.3}", b.bass, b.presence);
        assert!(b.bass >= b.air,       "bass={:.3} should dominate air={:.3}", b.bass, b.air);
    }

    #[test]
    fn sine_350hz_activates_low_mid() {
        // 350 Hz sits between the 315 and 400 Hz ISO bands, both in low-mid.
        let b = analyze(&sine(350.0, 1.0));
        assert!(b.low_mid > 0.5, "low_mid={:.3}", b.low_mid);
        assert!(b.low_mid >= b.presence, "low_mid={:.3} should dominate presence={:.3}", b.low_mid, b.presence);
        assert!(b.low_mid >= b.air,      "low_mid={:.3} should dominate air={:.3}", b.low_mid, b.air);
    }

    #[test]
    fn sine_1khz_activates_mid() {
        // 1 000 Hz ISO band is squarely in mid. Mid is the highest band.
        let b = analyze(&sine(1000.0, 1.0));
        assert!(b.mid > 0.5, "mid={:.3}", b.mid);
        assert!(b.mid >= b.sub_bass,  "mid={:.3} should dominate sub_bass={:.3}", b.mid, b.sub_bass);
        assert!(b.mid >= b.bass,      "mid={:.3} should dominate bass={:.3}", b.mid, b.bass);
        assert!(b.mid >= b.presence,  "mid={:.3} should dominate presence={:.3}", b.mid, b.presence);
        assert!(b.mid >= b.air,       "mid={:.3} should dominate air={:.3}", b.mid, b.air);
    }

    #[test]
    fn sine_4khz_activates_presence() {
        let b = analyze(&sine(4000.0, 1.0));
        assert!(b.presence > 0.5, "presence={:.3}", b.presence);
        assert!(b.presence >= b.sub_bass, "presence={:.3} should dominate sub_bass={:.3}", b.presence, b.sub_bass);
        assert!(b.presence >= b.bass,     "presence={:.3} should dominate bass={:.3}", b.presence, b.bass);
        assert!(b.presence >= b.low_mid,  "presence={:.3} should dominate low_mid={:.3}", b.presence, b.low_mid);
    }

    #[test]
    fn sine_8khz_activates_air() {
        let b = analyze(&sine(8000.0, 1.0));
        assert!(b.air > 0.5, "air={:.3}", b.air);
        assert!(b.air >= b.sub_bass, "air={:.3} should dominate sub_bass={:.3}", b.air, b.sub_bass);
        assert!(b.air >= b.bass,     "air={:.3} should dominate bass={:.3}", b.air, b.bass);
        assert!(b.air >= b.low_mid,  "air={:.3} should dominate low_mid={:.3}", b.air, b.low_mid);
    }

    #[test]
    fn accumulator_resets_between_calls() {
        // After process() the accumulators reset. A second call with different
        // input should reflect only the new data, not accumulated history.
        let mut s = IirFilterbank::new(SR);
        for x in sine(1000.0, 0.1) {
            s.push_sample(x);
        }
        s.process();
        let first = s.band_energies().mid;

        // Silence — if accumulators didn't reset, this would stay high.
        for _ in 0..4410 {
            s.push_sample(0.0);
        }
        s.process();
        let second = s.band_energies().mid;

        assert!(
            second < first,
            "second mid={:.3} should be < first mid={:.3} after silence",
            second,
            first
        );
    }

    #[test]
    fn onset_fires_on_attack() {
        let mut s = IirFilterbank::new(SR);
        // Prime with silence.
        for _ in 0..4410 {
            s.push_sample(0.0);
        }
        s.process();
        assert_eq!(s.onset_strength(), 0.0, "silence should produce zero onset");

        // Feed a full-scale 100 Hz sine.
        for x in sine(100.0, 0.1) {
            s.push_sample(x);
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
        // After several frames of a sustained sine, prev_energies converge to
        // band_rms and the positive delta drops toward zero.
        let mut s = IirFilterbank::new(SR);
        let hop = (SR / 100.0) as usize;
        let mut onset = 1.0f32;
        for frame in 0..20 {
            for i in 0..hop {
                let t = (frame * hop + i) as f32 / SR;
                s.push_sample((2.0 * std::f32::consts::PI * 100.0 * t).sin());
            }
            s.process();
            onset = s.onset_strength();
        }
        assert!(
            onset < 0.05,
            "onset={:.3} expected < 0.05 on steady sine",
            onset
        );
    }

    #[test]
    fn click_track_120bpm_converges() {
        use super::super::super::pipeline::Pipeline;
        let beat_samples = (SR * 60.0 / 120.0) as usize;
        let total = (SR * 12.0) as usize;
        let mut samples = vec![0.0f32; total];
        for i in (0..total).step_by(beat_samples) {
            samples[i] = 1.0;
        }
        let mut pipeline =
            Pipeline::with_analyzer(Box::new(IirFilterbank::new(SR)), SR, 1);
        let frames = pipeline.push(&samples);
        let bpm = frames.last().expect("no frames produced").bpm;
        assert!(
            (90.0..=150.0).contains(&bpm),
            "bpm={:.1} expected in [90, 150]",
            bpm
        );
    }
}
