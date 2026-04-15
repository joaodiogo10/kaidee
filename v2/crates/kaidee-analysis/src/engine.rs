use crate::analysis::run_analysis;
use crate::frame::AudioFrame;
use crate::stats::{Stats, StatsSnapshot};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use ringbuf::traits::{Producer, Split};
use ringbuf::HeapRb;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

const MAX_CHANNELS: usize = 2;
const RING_SIZE: usize = 44100 * 2 * MAX_CHANNELS;

/// Which audio input device to open.
pub enum DeviceSelection {
    /// OS default input device.
    Default,
    /// Select by position in the list returned by `list_input_devices()`.
    Index(usize),
    /// Select by name — partial, case-insensitive match.
    Name(String),
}

/// Returns all available input devices as `(index, name)` pairs.
/// The index can be passed directly to `DeviceSelection::Index`.
pub fn list_input_devices() -> Vec<(usize, String)> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devices) => devices
            .enumerate()
            .map(|(i, d)| (i, d.name().unwrap_or_else(|_| format!("device_{i}"))))
            .collect(),
        Err(e) => {
            eprintln!("could not enumerate input devices: {e}");
            vec![]
        }
    }
}

pub struct AnalysisEngine {
    _stream: cpal::Stream,
    shared: Arc<Mutex<AudioFrame>>,
    stats: Arc<Stats>,
}

impl AnalysisEngine {
    pub fn new(selection: DeviceSelection) -> Self {
        let device = open_device(selection);
        let config = device
            .default_input_config()
            .expect("no supported input config");

        println!(
            "using device: {}  ({} Hz, {} ch)",
            device.name().unwrap_or_default(),
            config.sample_rate().0,
            config.channels()
        );

        let sample_rate = config.sample_rate().0 as f32;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();
        let stream_config: cpal::StreamConfig = config.into();

        let (tx, rx) = HeapRb::<f32>::new(RING_SIZE).split();

        let shared = Arc::new(Mutex::new(AudioFrame::default()));
        let shared_writer = Arc::clone(&shared);

        let stats = Arc::new(Stats::new(sample_rate as u32));
        let stats_for_analysis = Arc::clone(&stats);
        let stats_for_callback = Arc::clone(&stats);

        std::thread::spawn(move || {
            run_analysis(rx, shared_writer, stats_for_analysis, sample_rate, channels);
        });

        let stream = if sample_format == SampleFormat::F32 {
            build_f32_stream(&device, stream_config, tx, stats_for_callback)
        } else if sample_format == SampleFormat::I16 {
            build_i16_stream(&device, stream_config, tx, stats_for_callback)
        } else {
            panic!("unsupported sample format: {:?}", sample_format)
        };

        stream.play().expect("failed to start audio stream");

        Self {
            _stream: stream,
            shared,
            stats,
        }
    }

    pub fn latest_frame(&self) -> AudioFrame {
        self.shared.lock().unwrap().clone()
    }

    pub fn stats_snapshot(&self) -> StatsSnapshot {
        self.stats.snapshot()
    }
}

fn open_device(selection: DeviceSelection) -> cpal::Device {
    let host = cpal::default_host();
    match selection {
        DeviceSelection::Default => host
            .default_input_device()
            .expect("no default input device found"),

        DeviceSelection::Index(idx) => host
            .input_devices()
            .expect("could not enumerate input devices")
            .nth(idx)
            .unwrap_or_else(|| panic!("no input device at index {idx}")),

        DeviceSelection::Name(name) => {
            let name_lower = name.to_lowercase();
            host.input_devices()
                .expect("could not enumerate input devices")
                .find(|d| {
                    d.name()
                        .map(|n| n.to_lowercase().contains(&name_lower))
                        .unwrap_or(false)
                })
                .unwrap_or_else(|| panic!("no input device matching '{name}'"))
        }
    }
}

fn build_f32_stream(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    mut tx: impl Producer<Item = f32> + Send + 'static,
    stats: Arc<Stats>,
) -> cpal::Stream {
    device
        .build_input_stream(
            &config,
            move |data: &[f32], _| {
                let pushed = tx.push_slice(data);
                if pushed < data.len() {
                    stats
                        .dropped_samples
                        .fetch_add((data.len() - pushed) as u64, Ordering::Relaxed);
                }
            },
            |e| eprintln!("audio error: {e}"),
            None,
        )
        .expect("failed to build F32 input stream")
}

fn build_i16_stream(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    mut tx: impl Producer<Item = f32> + Send + 'static,
    stats: Arc<Stats>,
) -> cpal::Stream {
    device
        .build_input_stream(
            &config,
            move |data: &[i16], _| {
                let f32_data: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                let pushed = tx.push_slice(&f32_data);
                if pushed < f32_data.len() {
                    stats
                        .dropped_samples
                        .fetch_add((f32_data.len() - pushed) as u64, Ordering::Relaxed);
                }
            },
            |e| eprintln!("audio error: {e}"),
            None,
        )
        .expect("failed to build I16 input stream")
}
