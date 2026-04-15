use kaidee_analysis::frame::AudioFrame;
use kaidee_analysis::stats::StatsSnapshot;

/// Abstraction over any display backend.
///
/// `TerminalUi` implements this now.
pub trait Visualizer {
    fn update(&mut self, frame: &AudioFrame, stats: &StatsSnapshot);
    fn render(&mut self);
}
