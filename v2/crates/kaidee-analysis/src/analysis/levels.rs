//! Amplitude-to-level mapping shared across spectral and dynamics.

/// Map a linear amplitude to 0.0–1.0 using a log (dB, decibel) scale.
/// Floor is -80 dB — mic input at higher frequencies is naturally quieter
/// and would clip to 0 with a -60 dB floor.
///   -80 dB → 0.0  (near silence)
///     0 dB → 1.0  (full scale)
pub fn to_01(linear: f32) -> f32 {
    let db = 20.0 * linear.max(1e-10).log10();
    ((db + 80.0) / 80.0).clamp(0.0, 1.0)
}
