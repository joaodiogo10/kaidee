//! Analysis thread — pops samples from the ring buffer and drives
//! [`Pipeline`], publishing each new [`AudioFrame`] to the shared state.

mod beat;
mod dynamics;
mod levels;
mod pipeline;
pub mod spectral;

pub use pipeline::Pipeline;

use crate::frame::AudioFrame;
use crate::global::ANALYSIS_RATE_HZ;
use crate::stats::Stats;
use ringbuf::traits::{Consumer, Observer};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub fn run_analysis(
    mut rx: impl Consumer<Item = f32> + Observer,
    shared: Arc<Mutex<AudioFrame>>,
    stats: Arc<Stats>,
    sample_rate: f32,
    channels: usize,
) {
    let hop_budget_micros = (1_000_000.0 / ANALYSIS_RATE_HZ) as u32;
    stats
        .hop_budget_micros
        .store(hop_budget_micros, Ordering::Relaxed);

    let mut pipeline = Pipeline::new(sample_rate, channels);
    let mut pop_buf = vec![0.0f32; 512 * channels];

    loop {
        let n = rx.pop_slice(&mut pop_buf);

        if n == 0 {
            std::thread::sleep(std::time::Duration::from_micros(500));
            continue;
        }

        let frame_start = Instant::now();
        let frames = pipeline.push(&pop_buf[..n]);

        if let Some(latest) = frames.last() {
            *shared.lock().unwrap() = latest.clone();
            let process_micros = frame_start.elapsed().as_micros() as u32;
            let occupancy = rx.occupied_len() as u32;
            stats.record_frame(process_micros, occupancy);
        }
    }
}
