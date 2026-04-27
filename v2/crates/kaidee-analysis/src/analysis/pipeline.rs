//! [`Pipeline`] — sample-source-agnostic analysis driver.

use super::beat::BeatTracker;
use super::dynamics;
use super::spectral::{FftAnalyzer, SpectralAnalyzer};
use super::ANALYSIS_RATE_HZ;
use crate::frame::AudioFrame;

/// Self-contained analysis pipeline.
///
/// Accepts interleaved audio samples via [`Pipeline::push`] and returns one
/// [`AudioFrame`] per analysis hop completed (~100 Hz at 44 100 Hz sample
/// rate). Stateless with respect to I/O — no ring buffer, no threads — so it
/// can be driven by any sample source: a live audio callback, a WAV file, or
/// a synthesized test signal.
///
/// The spectral analysis strategy is swappable at construction time via
/// [`Pipeline::with_analyzer`]. Defaults to [`FftAnalyzer`].
pub struct Pipeline {
    spectral: Box<dyn SpectralAnalyzer>,
    tracker: BeatTracker,
    samples_since_analysis: usize,
    hop: usize,
    channels: usize,
}

impl Pipeline {
    pub fn new(sample_rate: f32, channels: usize) -> Self {
        Self::with_analyzer(
            Box::new(FftAnalyzer::new(sample_rate)),
            sample_rate,
            channels,
        )
    }

    /// Construct a pipeline with a custom spectral analysis strategy.
    ///
    /// ```ignore
    /// let pipeline = Pipeline::with_analyzer(
    ///     Box::new(MyAnalyzer::new(sample_rate)),
    ///     sample_rate,
    ///     channels,
    /// );
    /// ```
    pub fn with_analyzer(
        analyzer: Box<dyn SpectralAnalyzer>,
        sample_rate: f32,
        channels: usize,
    ) -> Self {
        let hop = (sample_rate / ANALYSIS_RATE_HZ) as usize;
        let tracker = BeatTracker::new();
        Self {
            spectral: analyzer,
            tracker,
            samples_since_analysis: 0,
            hop,
            channels,
        }
    }

    /// Push interleaved audio samples and return one [`AudioFrame`] per
    /// analysis hop completed. Returns an empty vec if fewer than `hop`
    /// samples have accumulated since the last frame.
    ///
    /// Push and process are interleaved: `process()` fires every `hop` samples
    /// so each frame sees a fresh slice of audio. Calling `push` with a large
    /// batch (e.g. an entire WAV file) is equivalent to calling it in `hop`-
    /// sized chunks — the output is identical.
    pub fn push(&mut self, samples: &[f32]) -> Vec<AudioFrame> {
        let mut frames = Vec::new();
        for chunk in samples.chunks(self.channels) {
            let mono = chunk.iter().sum::<f32>() / self.channels as f32;
            self.spectral.push_sample(mono);
            self.samples_since_analysis += 1;

            if self.samples_since_analysis >= self.hop {
                self.samples_since_analysis -= self.hop;

                self.spectral.process();
                let onset: f32 = self.spectral.onset_strength();
                let bands = self.spectral.band_energies();
                let mags = self.spectral.magnitudes();
                let dyn_frame = dynamics::compute(mags);
                let (bpm, beat_phase, onset_strength) = self.tracker.update(onset);

                frames.push(AudioFrame {
                    sub_bass: bands.sub_bass,
                    bass: bands.bass,
                    low_mid: bands.low_mid,
                    mid: bands.mid,
                    presence: bands.presence,
                    air: bands.air,
                    rms: dyn_frame.rms,
                    peak: dyn_frame.peak,
                    bpm,
                    beat_phase,
                    onset_strength,
                    timestamp_ms: 0,
                });
            }
        }
        frames
    }
}
