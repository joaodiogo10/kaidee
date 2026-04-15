use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

/// Real-time pipeline health counters.
///
/// Shared via `Arc<Stats>` between the audio callback (writes dropped samples)
/// and the analysis thread (writes frame time / count / ring occupancy).
/// External consumers read via [`Stats::snapshot`], which returns a plain
/// [`StatsSnapshot`] value — `Arc<Stats>` itself stays inside the engine.
///
/// All writes are `Relaxed` — these are monitoring counters, not synchronization
/// primitives. [`Stats::record_frame`] does a non-atomic load-compute-store for
/// the moving average, which is safe because only the analysis thread ever
/// writes it.
pub struct Stats {
    /// Cumulative samples the audio callback tried to push but couldn't fit
    /// into the ring buffer. Non-zero means analysis is falling behind and
    /// audio is being lost.
    pub dropped_samples: AtomicU64,

    /// Per-frame processing time in microseconds (analysis thread), smoothed
    /// with an exponentially weighted moving average. Compare against
    /// `hop_budget_micros` — if close to or above, analysis is saturating
    /// the CPU.
    pub frame_process_micros: AtomicU32,

    /// Hop budget: the target time between analysis frames. Frame processing
    /// must stay below this to maintain real-time.
    pub hop_budget_micros: AtomicU32,

    /// Cumulative number of analysis frames
    pub frames_processed: AtomicU64,

    /// Current ring buffer occupancy (samples), sampled once per analysis
    /// frame. Healthy: near zero. Climbing: analysis is lagging.
    pub ring_occupancy: AtomicU32,

    /// When the engine started. Used to compute running averages.
    pub started_at: Instant,

    pub sample_rate: AtomicU32,
}

impl Stats {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            dropped_samples: AtomicU64::new(0),
            frame_process_micros: AtomicU32::new(0),
            hop_budget_micros: AtomicU32::new(0),
            frames_processed: AtomicU64::new(0),
            ring_occupancy: AtomicU32::new(0),
            started_at: Instant::now(),
            sample_rate: AtomicU32::new(sample_rate),
        }
    }

    /// Update the smoothed frame processing time with a new measurement.
    /// Uses an exponentially weighted moving average (α = 0.1).
    /// **Must only be called from the analysis thread** (single-writer).
    pub fn record_frame(&self, process_micros: u32, ring_occupancy: u32) {
        let prev = self.frame_process_micros.load(Ordering::Relaxed);
        let next = if prev == 0 {
            process_micros
        } else {
            // α = 0.1 — smooth enough to read, responsive enough to show spikes
            ((prev as f32) * 0.9 + (process_micros as f32) * 0.1) as u32
        };
        self.frame_process_micros.store(next, Ordering::Relaxed);
        self.ring_occupancy.store(ring_occupancy, Ordering::Relaxed);
        self.frames_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Running frames-per-second throughput since engine start.
    pub fn fps(&self) -> f32 {
        let frames = self.frames_processed.load(Ordering::Relaxed) as f32;
        let elapsed = self.started_at.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            frames / elapsed
        } else {
            0.0
        }
    }

    /// Plain-value snapshot of every counter in one consistent read pass.
    /// Consumers should prefer this over reading individual atomics — it avoids
    /// exposing `Arc<Stats>` outside the engine and gives display code pure
    /// values to work with.
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            dropped_samples: self.dropped_samples.load(Ordering::Relaxed),
            frame_process_micros: self.frame_process_micros.load(Ordering::Relaxed),
            hop_budget_micros: self.hop_budget_micros.load(Ordering::Relaxed),
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            ring_occupancy: self.ring_occupancy.load(Ordering::Relaxed),
            fps: self.fps(),
            sample_rate: self.sample_rate.load(Ordering::Relaxed),
        }
    }
}

/// Value-type snapshot of `Stats` at one point in time.
/// Produced by `Stats::snapshot()` and passed into the visualizer each tick.
#[derive(Debug, Clone, Default)]
pub struct StatsSnapshot {
    pub dropped_samples: u64,
    pub frame_process_micros: u32,
    pub hop_budget_micros: u32,
    pub frames_processed: u64,
    pub ring_occupancy: u32,
    pub fps: f32,
    pub sample_rate: u32,
}

impl Default for Stats {
    fn default() -> Self {
        Self::new(0)
    }
}
