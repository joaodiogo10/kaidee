mod terminal_ui;
mod visualizer;

use clap::Parser;
use kaidee_analysis::engine::{AnalysisEngine, DeviceSelection, list_input_devices};
use terminal_ui::TerminalUi;
use visualizer::Visualizer;

#[derive(Parser)]
#[command(name = "kaidee", about = "Audio-reactive visual engine")]
struct Args {
    /// List available audio input devices and exit
    #[arg(long)]
    list_devices: bool,

    /// Audio input device — accepts an index (e.g. 2) or a name fragment
    /// (e.g. "BlackHole"). Run --list-devices to see available options.
    #[arg(long)]
    device: Option<String>,
}

fn main() {
    let args = Args::parse();

    if args.list_devices {
        let devices = list_input_devices();
        if devices.is_empty() {
            println!("no input devices found");
        } else {
            println!("available input devices:");
            for (i, name) in devices {
                println!("  {i}: {name}");
            }
        }
        return;
    }

    // Parse --device: a bare number → index, anything else → name fragment
    let selection = match args.device {
        None => DeviceSelection::Default,
        Some(s) => match s.parse::<usize>() {
            Ok(idx) => DeviceSelection::Index(idx),
            Err(_) => DeviceSelection::Name(s),
        },
    };

    let engine = AnalysisEngine::new(selection);
    let mut ui: Box<dyn Visualizer> = Box::new(TerminalUi::new());

    loop {
        let frame = engine.latest_frame();
        let snapshot = engine.stats_snapshot();
        ui.update(&frame, &snapshot);
        ui.render();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
