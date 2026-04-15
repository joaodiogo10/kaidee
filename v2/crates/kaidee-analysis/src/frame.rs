/// One snapshot of audio analysis results, produced on every analysis tick.
///
/// This is the public API boundary between the analysis engine and all consumers
/// (terminal UI, GPU renderer, etc.). Consumers never touch raw audio or FFT data.
///
/// Fields will be populated as each analysis layer is implemented. Unimplemented
/// fields default to 0.0.
#[derive(Debug, Clone, Default)]
pub struct AudioFrame {
    pub timestamp_ms: u64,

    // --- Band energies (0.0–1.0) ---
    pub sub_bass: f32,  // < 60 Hz
    pub bass: f32,      // 60–250 Hz
    pub low_mid: f32,   // 250–500 Hz
    pub mid: f32,       // 500–2000 Hz
    pub presence: f32,  // 2000–6000 Hz
    pub air: f32,       // > 6000 Hz

    // --- Rhythm ---
    pub bpm: f32,
    pub beat_phase: f32,      // 0.0–1.0 position within current beat
    pub onset_strength: f32,  // 0.0–1.0

    // --- Dynamics ---
    pub rms: f32,
    pub peak: f32,
}
