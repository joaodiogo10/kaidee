//! Analysis thread — pulls samples from the ring buffer, schedules FFT hops,
//! and fans out to the algorithm blocks (spectral, dynamics, beat) before
//! publishing a combined [`AudioFrame`].

mod beat;
mod dynamics;
mod spectral;

use crate::frame::AudioFrame;
use crate::stats::Stats;
use beat::BeatTracker;
use ringbuf::traits::{Consumer, Observer};
use spectral::Spectral;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub(crate) const ANALYSIS_RATE_HZ: f32 = 100.0;

/// Map a linear amplitude to 0.0–1.0 using a log (dB, decibel) scale.
/// Floor is -80 dB — mic input at higher frequencies is naturally quieter
/// and would clip to 0 with a -60 dB floor.
///   -80 dB → 0.0  (near silence)
///     0 dB → 1.0  (full scale)
pub(super) fn to_01(linear: f32) -> f32 {
    let db = 20.0 * linear.max(1e-10).log10();
    ((db + 80.0) / 80.0).clamp(0.0, 1.0)
}

pub fn run_analysis(
    mut rx: impl Consumer<Item = f32> + Observer,
    shared: Arc<Mutex<AudioFrame>>,
    stats: Arc<Stats>,
    sample_rate: f32,
    channels: usize,
) {
    let hop = (sample_rate / ANALYSIS_RATE_HZ) as usize;
    let hop_budget_micros = (1_000_000.0 / ANALYSIS_RATE_HZ) as u32;
    stats
        .hop_budget_micros
        .store(hop_budget_micros, Ordering::Relaxed);

    let mut spectral = Spectral::new(sample_rate);
    let mut tracker = BeatTracker::new(sample_rate, hop, spectral.n_bins());

    let mut samples_since_analysis = 0usize;
    let mut pop_buf = vec![0.0f32; 512 * channels];

    loop {
        let n = rx.pop_slice(&mut pop_buf);

        // buffer empty
        if n == 0 {
            std::thread::sleep(std::time::Duration::from_micros(500));
            continue;
        }

        // Mix interleaved frames down to mono for the FFT window.
        for frame in pop_buf[..n].chunks(channels) {
            let mono = frame.iter().sum::<f32>() / channels as f32;
            spectral.push_sample(mono);
            samples_since_analysis += 1;
        }

        // Drain all accumulated hops in this pop batch.
        while samples_since_analysis >= hop {
            samples_since_analysis -= hop;

            let frame_start = Instant::now();

            spectral.process();
            let bands = spectral.band_energies();
            let mags = spectral.magnitudes();
            let dyn_frame = dynamics::compute(mags);
            let (bpm, beat_phase, onset_strength) = tracker.update(mags);

            let audio_frame = AudioFrame {
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
            };
            *shared.lock().unwrap() = audio_frame;

            let process_micros = frame_start.elapsed().as_micros() as u32;
            let occupancy = rx.occupied_len() as u32;
            stats.record_frame(process_micros, occupancy);
        }
    }
}
