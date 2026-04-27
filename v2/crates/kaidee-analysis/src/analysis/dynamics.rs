//! Time-domain-style dynamics features: overall RMS (root mean square)
//! energy and peak magnitude, both mapped to 0.0–1.0. Future expansion:
//! LUFS (loudness units relative to full scale), crest factor.

use super::levels::to_01;

pub struct DynamicsFrame {
    pub rms: f32,
    pub peak: f32,
}

pub fn compute(magnitudes: &[f32]) -> DynamicsFrame {
    let mean_sq: f32 =
        magnitudes.iter().map(|m| m * m).sum::<f32>() / magnitudes.len() as f32;
    let rms = mean_sq.sqrt();
    let peak = magnitudes.iter().cloned().fold(0.0f32, f32::max);

    DynamicsFrame {
        rms: to_01(rms),
        peak: to_01(peak),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::spectral::{FftAnalyzer, SpectralAnalyzer};

    const SR: f32 = 44100.0;

    fn analyze(samples: &[f32]) -> DynamicsFrame {
        let mut s = FftAnalyzer::new(SR);
        for &x in samples {
            s.push_sample(x);
        }
        s.process();
        compute(s.magnitudes())
    }

    fn sine(freq_hz: f32, duration_secs: f32) -> Vec<f32> {
        let n = (SR * duration_secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / SR).sin())
            .collect()
    }

    #[test]
    fn full_scale_sine_has_significant_rms() {
        // Spectral RMS is the mean of squared magnitudes across all 1 024 bins,
        // so a pure sine — which only activates one bin — reads much lower than
        // a broadband signal at the same peak amplitude. A full-scale 1 kHz sine
        // maps to roughly -36 dB spectral RMS, which is ~0.55 on the 0–1 scale.
        let frame = analyze(&sine(1000.0, 1.0));
        assert!(frame.rms > 0.4, "rms={:.3} expected > 0.4", frame.rms);
    }

    #[test]
    fn louder_signal_higher_rms() {
        let quiet = analyze(&sine(1000.0, 1.0).iter().map(|s| s * 0.01).collect::<Vec<_>>());
        let loud  = analyze(&sine(1000.0, 1.0));
        assert!(loud.rms > quiet.rms, "loud={:.3} quiet={:.3}", loud.rms, quiet.rms);
    }
}
