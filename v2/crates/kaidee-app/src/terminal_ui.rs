use crate::visualizer::Visualizer;
use kaidee_analysis::frame::AudioFrame;
use kaidee_analysis::stats::StatsSnapshot;

pub struct TerminalUi {
    latest: AudioFrame,
    stats: StatsSnapshot,
}

impl TerminalUi {
    pub fn new() -> Self {
        Self {
            latest: AudioFrame::default(),
            stats: StatsSnapshot::default(),
        }
    }
}

impl Visualizer for TerminalUi {
    fn update(&mut self, frame: &AudioFrame, stats: &StatsSnapshot) {
        self.latest = frame.clone();
        self.stats = stats.clone();
    }

    fn render(&mut self) {
        let f = &self.latest;

        // Move cursor to top-left and clear — avoids flicker vs full clear
        print!("\x1B[H\x1B[2J");

        println!("Kaidee v2\n");
        print_bar("SUB  ", f.sub_bass);
        print_bar("BASS ", f.bass);
        print_bar("L-MID", f.low_mid);
        print_bar("MID  ", f.mid);
        print_bar("PRES ", f.presence);
        print_bar("AIR  ", f.air);
        println!();

        let beat_filled = (f.beat_phase * 16.0) as usize;
        let beat_bar: String = (0..16)
            .map(|i| if i < beat_filled { '▮' } else { '░' })
            .collect();
        println!(
            "BPM {:.1}  beat [{}]  onset {:.2}",
            f.bpm, beat_bar, f.onset_strength
        );
        println!("RMS {:.3}  peak {:.3}", f.rms, f.peak);
        println!();
        self.render_stats();
    }
}

impl TerminalUi {
    fn render_stats(&self) {
        let s = &self.stats;
        let budget_us = s.hop_budget_micros.max(1);
        let usage_pct = (s.frame_process_micros as f32 / budget_us as f32) * 100.0;

        // ANSI colors: green / yellow / red based on CPU headroom and drops.
        // A healthy pipeline is <50% hop budget with zero drops.
        let usage_color = if usage_pct > 90.0 || s.dropped_samples > 0 {
            "\x1B[31m" // red
        } else if usage_pct > 60.0 {
            "\x1B[33m" // yellow
        } else {
            "\x1B[32m" // green
        };
        let reset = "\x1B[0m";

        println!("── pipeline ───────────────────────────");
        println!("Sample rate: {} hz", s.sample_rate);
        println!(
            "proc  {color}{:>5.2} ms{reset} / {:.1} ms  ({color}{:>5.1}%{reset})",
            s.frame_process_micros as f32 / 1000.0,
            budget_us as f32 / 1000.0,
            usage_pct,
            color = usage_color,
            reset = reset,
        );
        println!(
            "fps   {:>5.1} (target 100.0)  frames {}",
            s.fps, s.frames_processed
        );
        println!("ring  {} samples", s.ring_occupancy);
        let drops_color = if s.dropped_samples > 0 {
            "\x1B[31m"
        } else {
            "\x1B[32m"
        };
        println!(
            "drops {color}{}{reset} samples",
            s.dropped_samples,
            color = drops_color,
            reset = reset
        );
    }
}

fn print_bar(label: &str, value: f32) {
    let clamped = value.clamp(0.0, 1.0);
    let filled = (clamped * 40.0) as usize;
    let empty = 40usize.saturating_sub(filled);
    println!(
        "[{}] {}{} {:.2}",
        label,
        "█".repeat(filled),
        "░".repeat(empty),
        clamped
    );
}
