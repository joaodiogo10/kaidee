//! Spectral analysis strategies.
//!
//! [`SpectralAnalyzer`] defines the interface all strategies must implement.
//! Swap strategies at construction time via [`Pipeline::with_analyzer`].
//!
//! | Strategy        | Algorithm                        | Frequency resolution          |
//! |-----------------|----------------------------------|-------------------------------|
//! | [`FftAnalyzer`] | Short-time FFT with Hann window  | ~21.5 Hz/bin at 44 100 Hz     |
//!
//! ## Swapping strategies changes beat tracker performance
//!
//! The beat tracker consumes [`SpectralAnalyzer::onset_strength`] — a scalar
//! computed by each strategy using its own representation. Strategies with
//! lower latency or finer bass resolution produce sharper onset signals, which
//! directly affects BPM convergence speed and beat phase stability. This is
//! intentional. The beat tracker's tunable constants (onset threshold, PLL
//! (phase-locked loop) gain, BPM smoothing) were calibrated against
//! [`FftAnalyzer`] and may need re-tuning for other strategies.

mod fft;
pub use fft::FftAnalyzer;

// Alternative strategy, not yet wired into the default pipeline. Exposed
// as a public submodule so consumers can opt in via `Pipeline::with_analyzer`.
pub mod iir;

/// Per-band energy levels from one analysis frame, all mapped to 0.0–1.0.
pub struct BandEnergies {
    /// 20–60 Hz — sub-bass rumble, kick drum body.
    pub sub_bass: f32,
    /// 60–250 Hz — bass guitar, kick drum attack, low-end punch.
    pub bass: f32,
    /// 250–500 Hz — low mids, male vocal body, bass guitar harmonics.
    pub low_mid: f32,
    /// 500–2 000 Hz — core melody, snare body, most vocal fundamentals.
    pub mid: f32,
    /// 2 000–6 000 Hz — vocal presence, instrument attack, intelligibility.
    pub presence: f32,
    /// 6 000 Hz–Nyquist — air, cymbal shimmer, high-frequency brightness.
    pub air: f32,
}

/// Interface shared by all spectral analysis strategies.
///
/// Implementations receive mono samples via [`push_sample`], run their
/// analysis when [`process`] is called, then expose results via
/// [`magnitudes`], [`band_energies`], and [`onset_strength`].
pub trait SpectralAnalyzer {
    /// Push one mono sample into the analyser's internal buffer.
    fn push_sample(&mut self, sample: f32);

    /// Run analysis on the current buffer and update internal state.
    /// Must be called before [`magnitudes`], [`band_energies`], or
    /// [`onset_strength`].
    fn process(&mut self);

    /// Magnitude spectrum from the last [`process`] call. Length = [`n_bins`].
    fn magnitudes(&self) -> &[f32];

    /// Per-band energy levels from the last [`process`] call, all 0.0–1.0.
    fn band_energies(&self) -> BandEnergies;

    /// Onset strength from the last [`process`] call. Range 0.0–1.0.
    ///
    /// Measures how much the spectrum changed since the previous frame,
    /// weighted toward the bass region where kick drum transients live.
    fn onset_strength(&self) -> f32;

    /// Number of frequency bins in [`magnitudes`].
    fn n_bins(&self) -> usize;
}
