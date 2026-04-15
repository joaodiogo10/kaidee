//! Time-domain-style dynamics features: overall RMS (root mean square)
//! energy and peak magnitude, both mapped to 0.0–1.0. Future expansion:
//! LUFS (loudness units relative to full scale), crest factor.

use super::to_01;

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
