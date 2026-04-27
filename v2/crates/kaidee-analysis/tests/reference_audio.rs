//! Reference audio tests — two tiers:
//!
//! **Synthetic signals** (always run in CI): generated audio with known
//! spectral, dynamic, and rhythmic properties. Each test asserts a specific
//! field of [`AudioFrame`]. Together they cover every output the pipeline
//! produces.
//!
//! **WAV fixture tests** (ignored by default): load real audio from
//! `v2/fixtures/`. See [`FixtureSpec`] and `v2/fixtures/README.md` for
//! how to add files and what values to assert.

use kaidee_analysis::analysis::Pipeline;
use kaidee_analysis::frame::AudioFrame;

const SR: f32 = 44100.0;

// ---------------------------------------------------------------------------
// Reference fixture tolerance constants
//
// These define how closely Kaidee's output must match librosa's reference
// measurements. They live here — not in the JSON — because they express a
// property of Kaidee, not of the reference tool.
// ---------------------------------------------------------------------------

/// BPM tolerance for real-audio fixtures. Tempo estimation on real
/// recordings carries 1–2 BPM of uncertainty.
const BPM_TOL: f32 = 2.0;

/// Kaidee's band energy must reach at least this fraction of librosa's
/// measured share for each active band (one with > BAND_ACTIVE_THRESHOLD share).
const BAND_ENERGY_FLOOR: f32 = 0.5;

/// Bands below this share in the reference are considered inactive and are
/// not asserted (librosa's energy estimate is too noisy at very low levels).
const BAND_ACTIVE_THRESHOLD: f32 = 0.05;

/// Skip BPM assertions until this many seconds into the track to give the
/// beat tracker time to converge before the first per-segment check.
const CONVERGENCE_SECS: f32 = 10.0;

/// Kaidee's post-convergence BPM standard deviation (SD) must stay below
/// this multiple of librosa's measured per-segment BPM SD.
const BPM_STD_MAX_RATIO: f32 = 2.0;

/// Floor for the BPM SD limit — even a perfectly stable track gets at least
/// this much headroom.
const BPM_STD_FLOOR: f32 = 1.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run `samples` through the pipeline and return all produced frames.
fn run(samples: &[f32]) -> Vec<AudioFrame> {
    let mut pipeline = Pipeline::new(SR, 1);
    pipeline.push(samples)
}

/// Run `samples` and return the last produced frame.
fn last_frame(samples: &[f32]) -> AudioFrame {
    run(samples).into_iter().last().expect("no frames produced")
}

/// Average a field over the last `tail_secs` seconds of frames.
/// Useful for checking steady-state values after initial transients.
fn mean_tail(frames: &[AudioFrame], tail_secs: f32, f: impl Fn(&AudioFrame) -> f32) -> f32 {
    let n_tail = (tail_secs * 100.0) as usize; // 100 Hz analysis rate
    let tail = if frames.len() > n_tail {
        &frames[frames.len() - n_tail..]
    } else {
        frames
    };
    tail.iter().map(|fr| f(fr)).sum::<f32>() / tail.len() as f32
}

/// Generate a pure sine at `freq_hz` for `duration_secs`.
fn sine(freq_hz: f32, duration_secs: f32) -> Vec<f32> {
    let n = (SR * duration_secs) as usize;
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / SR).sin())
        .collect()
}

/// Generate silence.
fn silence(duration_secs: f32) -> Vec<f32> {
    vec![0.0f32; (SR * duration_secs) as usize]
}

/// One kick drum hit: decaying sine at `freq_hz` with amplitude envelope
/// `exp(-decay × t)`.
fn kick_drum(freq_hz: f32, decay: f32) -> Vec<f32> {
    let duration = (5.0 / decay).min(0.15);
    let n = (SR * duration) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SR;
            (-decay * t).exp() * (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

/// 4/4 kick-on-every-beat pattern using a 60 Hz decaying-sine kick.
fn drum_pattern(bpm: f32, duration_secs: f32) -> Vec<f32> {
    let beat_samples = (SR * 60.0 / bpm) as usize;
    let total = (SR * duration_secs) as usize;
    let kick = kick_drum(60.0, 30.0);
    let mut out = vec![0.0f32; total];
    for beat_start in (0..total).step_by(beat_samples) {
        let copy_len = kick.len().min(total - beat_start);
        for (j, &s) in kick[..copy_len].iter().enumerate() {
            out[beat_start + j] += s;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Frequency band tests
// ---------------------------------------------------------------------------

#[test]
fn silence_all_bands_zero() {
    let fr = last_frame(&silence(1.0));
    assert_eq!(fr.sub_bass, 0.0, "sub_bass");
    assert_eq!(fr.bass, 0.0, "bass");
    assert_eq!(fr.low_mid, 0.0, "low_mid");
    assert_eq!(fr.mid, 0.0, "mid");
    assert_eq!(fr.presence, 0.0, "presence");
    assert_eq!(fr.air, 0.0, "air");
}

#[test]
fn sub_bass_sine_dominates_low_bands() {
    // 40 Hz is squarely in sub-bass (< 60 Hz). Bass picks up spillover from
    // FFT leakage; mid and above should be negligible.
    let frames = run(&sine(40.0, 2.0));
    let sub = mean_tail(&frames, 1.0, |f| f.sub_bass);
    let mid = mean_tail(&frames, 1.0, |f| f.mid);
    assert!(sub > 0.8, "sub_bass={sub:.3} expected > 0.8");
    assert!(mid < 0.1, "mid={mid:.3} expected < 0.1");
}

#[test]
fn bass_sine_dominates() {
    // 150 Hz is inside the bass band (60–250 Hz).
    let frames = run(&sine(150.0, 2.0));
    let bass = mean_tail(&frames, 1.0, |f| f.bass);
    let presence = mean_tail(&frames, 1.0, |f| f.presence);
    assert!(bass > 0.8, "bass={bass:.3} expected > 0.8");
    assert!(presence < 0.05, "presence={presence:.3} expected < 0.05");
}

#[test]
fn mid_sine_dominates() {
    // 1 kHz is inside mid (500–2000 Hz). Sub-bass and bass should be zero.
    let frames = run(&sine(1000.0, 2.0));
    let mid = mean_tail(&frames, 1.0, |f| f.mid);
    let sub = mean_tail(&frames, 1.0, |f| f.sub_bass);
    let bass = mean_tail(&frames, 1.0, |f| f.bass);
    assert!(mid > 0.8, "mid={mid:.3} expected > 0.8");
    assert_eq!(sub, 0.0, "sub_bass should be zero for 1 kHz");
    assert_eq!(bass, 0.0, "bass should be zero for 1 kHz");
}

#[test]
fn presence_sine_dominates() {
    // 3 kHz is inside presence (2000–6000 Hz).
    let frames = run(&sine(3000.0, 2.0));
    let presence = mean_tail(&frames, 1.0, |f| f.presence);
    let bass = mean_tail(&frames, 1.0, |f| f.bass);
    assert!(presence > 0.8, "presence={presence:.3} expected > 0.8");
    assert_eq!(bass, 0.0, "bass should be zero for 3 kHz");
}

#[test]
fn air_sine_dominates() {
    // 10 kHz is inside air (> 6000 Hz).
    let frames = run(&sine(10_000.0, 2.0));
    let air = mean_tail(&frames, 1.0, |f| f.air);
    let sub = mean_tail(&frames, 1.0, |f| f.sub_bass);
    assert!(air > 0.8, "air={air:.3} expected > 0.8");
    assert_eq!(sub, 0.0, "sub_bass should be zero for 10 kHz");
}

// ---------------------------------------------------------------------------
// Dynamics tests (RMS, peak)
// ---------------------------------------------------------------------------

#[test]
fn silence_rms_and_peak_zero() {
    let fr = last_frame(&silence(1.0));
    assert_eq!(fr.rms, 0.0, "rms");
    assert_eq!(fr.peak, 0.0, "peak");
}

#[test]
fn full_scale_sine_rms_near_one() {
    // RMS is computed over FFT magnitudes (spectral RMS), not time-domain
    // samples. A pure sine activates only one bin out of 1 024, so spectral
    // RMS is much lower than 1/√2. A full-scale 440 Hz sine typically reads
    // ~0.55 on the 0–1 scale — noticeably above silence (0.0) and above a
    // half-amplitude sine (~0.40). "Near one" is a misnomer; the assertion
    // is really "well above zero for a full-scale tone".
    let frames = run(&sine(440.0, 2.0));
    let rms = mean_tail(&frames, 1.0, |f| f.rms);
    assert!(rms > 0.5, "rms={rms:.3} expected > 0.5 for full-scale sine");
}

#[test]
fn half_scale_sine_rms_lower_than_full() {
    // A half-amplitude sine should produce a noticeably lower RMS than
    // a full-scale sine.
    let half: Vec<f32> = sine(440.0, 2.0).iter().map(|&s| s * 0.5).collect();
    let full = sine(440.0, 2.0);
    let frames_half = run(&half);
    let frames_full = run(&full);
    let rms_half = mean_tail(&frames_half, 1.0, |f| f.rms);
    let rms_full = mean_tail(&frames_full, 1.0, |f| f.rms);
    assert!(
        rms_half < rms_full,
        "half-scale rms {rms_half:.3} should be less than full-scale {rms_full:.3}"
    );
}

#[test]
fn peak_at_least_rms() {
    // Peak ≥ RMS always holds for any signal.
    let frames = run(&sine(440.0, 2.0));
    let rms = mean_tail(&frames, 1.0, |f| f.rms);
    let peak = mean_tail(&frames, 1.0, |f| f.peak);
    assert!(peak >= rms, "peak {peak:.3} should be >= rms {rms:.3}");
}

// ---------------------------------------------------------------------------
// Onset strength tests
// ---------------------------------------------------------------------------

#[test]
fn silence_onset_zero() {
    let fr = last_frame(&silence(1.0));
    assert_eq!(fr.onset_strength, 0.0);
}

#[test]
fn onset_fires_on_kick_attack() {
    // Feed silence, then a single kick. The frame immediately after the
    // kick should have a positive onset_strength.
    let n_silence = (SR * 0.5) as usize; // 0.5 s pre-roll
    let kick = kick_drum(60.0, 30.0);
    let mut audio = vec![0.0f32; n_silence];
    audio.extend_from_slice(&kick);
    audio.extend(silence(0.2).iter()); // short tail so we see the transition

    let frames = run(&audio);
    // Find the maximum onset in the last 200 ms (kick region)
    let n_tail = 20; // 200 ms at 100 Hz
    let max_onset = frames
        .iter()
        .rev()
        .take(n_tail)
        .map(|f| f.onset_strength)
        .fold(0.0f32, f32::max);
    // The kick is a short 60 Hz decaying sine. FftAnalyzer computes spectral
    // flux (sum of positive magnitude deltas across all bins), so a single
    // narrow-band transient produces a small but non-zero onset. 0.01 is a
    // conservative floor — real kick drums with broadband attack content
    // produce much higher values in practice.
    assert!(
        max_onset > 0.01,
        "onset_strength={max_onset:.3} expected > 0.01 after kick attack"
    );
}

#[test]
fn onset_low_during_steady_sine() {
    // After several frames of a sustained sine, positive flux → 0.
    let frames = run(&sine(440.0, 3.0));
    let onset = mean_tail(&frames, 1.0, |f| f.onset_strength);
    assert!(
        onset < 0.05,
        "onset_strength={onset:.3} expected < 0.05 during steady sine"
    );
}

// ---------------------------------------------------------------------------
// BPM and beat phase tests
// ---------------------------------------------------------------------------

#[test]
fn drum_machine_120bpm_converges() {
    // Realistic kick drum waveform instead of a delta impulse — verifies
    // onset detection responds to actual sub-bass content.
    let frames = run(&drum_pattern(120.0, 12.0));
    let bpm = frames.last().unwrap().bpm;
    assert!(
        (119.5..=120.5).contains(&bpm),
        "120 BPM drum: got {bpm:.2}"
    );
}

#[test]
fn drum_machine_128bpm_converges() {
    // 128 BPM is not in the unit-test suite. Lag ≈ 46.88 frames — tests
    // parabolic interpolation at a non-round tempo.
    let frames = run(&drum_pattern(128.0, 14.0));
    let bpm = frames.last().unwrap().bpm;
    assert!(
        (127.5..=128.5).contains(&bpm),
        "128 BPM drum: got {bpm:.2}"
    );
}

#[test]
fn drum_machine_140bpm_converges() {
    let frames = run(&drum_pattern(140.0, 16.0));
    let bpm = frames.last().unwrap().bpm;
    assert!(
        (139.5..=140.5).contains(&bpm),
        "140 BPM drum: got {bpm:.2}"
    );
}

#[test]
fn beat_phase_cycles_zero_to_one() {
    // After BPM convergence the beat_phase should complete full 0→1 cycles.
    // Check that over a 2-beat window we see both low (< 0.1) and high (> 0.9)
    // phase values — confirming the PLL is locked and cycling.
    let frames = run(&drum_pattern(120.0, 16.0));
    // Use the last 2 seconds (2 beats at 120 BPM) for phase inspection.
    let tail: Vec<f32> = frames
        .iter()
        .rev()
        .take(200) // 2 s at 100 Hz
        .map(|f| f.beat_phase)
        .collect();
    let has_low = tail.iter().any(|&p| p < 0.1);
    let has_high = tail.iter().any(|&p| p > 0.9);
    assert!(has_low, "beat_phase never reached < 0.1 — PLL may not be cycling");
    assert!(has_high, "beat_phase never reached > 0.9 — PLL may not be cycling");
}

// ---------------------------------------------------------------------------
// Temporal continuity tests (synthetic)
// ---------------------------------------------------------------------------

#[test]
fn bpm_stays_stable_after_convergence() {
    // After ~12 s of convergence, BPM should stay within ±0.5 of the target
    // for the remaining duration. A high standard deviation (SD) here means
    // the tracker is drifting or oscillating — not just slow to converge.
    let frames = run(&drum_pattern(128.0, 30.0));

    // Skip the first 12 s (1 200 frames) — convergence window.
    let post_convergence = &frames[1200.min(frames.len())..];
    assert!(
        !post_convergence.is_empty(),
        "not enough frames to check post-convergence stability"
    );

    let bpms: Vec<f32> = post_convergence.iter().map(|f| f.bpm).collect();
    let mean = bpms.iter().sum::<f32>() / bpms.len() as f32;
    let std = (bpms.iter().map(|b| (b - mean).powi(2)).sum::<f32>() / bpms.len() as f32).sqrt();

    assert!(
        std < 0.5,
        "BPM SD after convergence: {std:.3} — tracker is drifting (mean={mean:.2})"
    );
}

#[test]
fn beat_phase_advances_smoothly() {
    // After BPM convergence the phase-locked loop (PLL) should advance the
    // beat_phase by approximately bpm/60/100 ≈ 0.021 per frame at 128 BPM.
    //
    // We check that no single inter-frame phase jump (unwrapped) deviates from
    // the expected advance by more than ±0.05. A larger jump means the PLL
    // made a sudden correction — acceptable during lock-on, not after.
    let frames = run(&drum_pattern(128.0, 30.0));

    // Skip the first 15 s — allow time for BPM and phase to stabilise.
    let stable = &frames[1500.min(frames.len())..];
    assert!(!stable.is_empty(), "not enough frames for phase continuity check");

    let expected_advance = 128.0_f32 / 60.0 / 100.0; // ≈ 0.0213 per frame

    let mut violations = 0usize;
    for window in stable.windows(2) {
        let prev = window[0].beat_phase;
        let curr = window[1].beat_phase;

        // Unwrap across the 0→1 boundary.
        let raw_delta = curr - prev;
        let delta = if raw_delta < -0.5 {
            raw_delta + 1.0 // wrapped through 0
        } else if raw_delta > 0.5 {
            raw_delta - 1.0 // wrapped backwards (shouldn't happen in normal operation)
        } else {
            raw_delta
        };

        if (delta - expected_advance).abs() > 0.05 {
            violations += 1;
        }
    }

    // Allow at most 1% of frames to have a large correction — occasional PLL
    // nudges from strong onsets are expected even in the stable region.
    let max_violations = (stable.len() as f32 * 0.01) as usize + 1;
    assert!(
        violations <= max_violations,
        "beat_phase had {violations} large jumps (allowed {max_violations}) — PLL corrections are too frequent"
    );
}

// ---------------------------------------------------------------------------
// WAV fixture tests
// ---------------------------------------------------------------------------

/// Expected output ranges for a reference audio file.
/// All fields are `Option` — `None` means "do not assert this field."
///
/// Values are checked against the **mean of the last 10 seconds** of frames,
/// except for `bpm` which is checked on the final frame after convergence.
#[derive(Default)]
struct FixtureSpec {
    /// Expected BPM, asserted on the last frame.
    pub bpm: Option<f32>,
    /// Tolerance around `bpm`. Defaults to ±0.5 for synthetic/clean signals.
    /// For JSON-backed fixtures the tolerance is computed from tool spread
    /// by `bpm_tolerance_from_tools` — this field is only used by `FIXTURES`.
    pub bpm_tolerance: f32,
    /// Minimum mean value for each band over the tail window.
    pub sub_bass_min: Option<f32>,
    pub bass_min: Option<f32>,
    pub low_mid_min: Option<f32>,
    pub mid_min: Option<f32>,
    pub presence_min: Option<f32>,
    pub air_min: Option<f32>,
    /// Which band should have the highest mean energy.
    pub dominant_band: Option<&'static str>,
    /// Mean RMS range over tail window.
    pub rms_min: Option<f32>,
    pub rms_max: Option<f32>,
    /// Onset strength: at least one frame in the tail should exceed this.
    pub onset_min_peak: Option<f32>,
}

impl FixtureSpec {
    /// Default BPM tolerance for hand-written specs: ±0.5 (same as unit tests).
    fn effective_bpm_tolerance(&self) -> f32 {
        if self.bpm_tolerance == 0.0 { 0.5 } else { self.bpm_tolerance }
    }
}

/// Decode a WAV or MP3 file to a mono f32 sample buffer.
/// Stereo (and higher) inputs are mixed down by averaging channels.
/// Returns `None` if the file does not exist or cannot be decoded.
fn load_audio_mono(path: &str) -> Option<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .ok()?;

    let mut format = probed.format;
    let track = format.default_track()?;
    let sample_rate = track.codec_params.sample_rate?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .ok()?;

    let mut interleaved: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(_)) | Err(Error::ResetRequired) => break,
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let spec = *decoded.spec();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        interleaved.extend_from_slice(buf.samples());
    }

    if interleaved.is_empty() {
        return None;
    }

    let mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Some((mono, sample_rate))
}

/// Add an entry here for each WAV file you place in `../../fixtures/` that
/// you want to assert with hand-written values. The test is gated behind
/// `#[ignore]` — run with `-- --include-ignored`.
///
/// For a fully automated workflow, prefer the `reference_json_fixtures` test:
/// run `python v2/tools/analyze_reference.py ../../fixtures/` once after
/// adding WAV files and the JSON files are created automatically.
///
/// Example entry (uncomment and adapt when you have a file):
/// ```
/// ("../../fixtures/128bpm_techno.wav", FixtureSpec {
///     bpm: Some(128.0),
///     bass_min: Some(0.3),
///     dominant_band: Some("bass"),
///     rms_min: Some(0.1),
///     onset_min_peak: Some(0.2),
///     ..Default::default()
/// }),
/// ```
const FIXTURES: &[(&str, fn() -> FixtureSpec)] = &[
    // -- add entries here --
];

#[test]
#[ignore = "requires WAV fixtures — see ../../fixtures/README.md"]
fn reference_wav_fixtures() {
    assert!(
        !FIXTURES.is_empty(),
        "No fixtures registered. Add entries to FIXTURES in reference_audio.rs."
    );
    for (path, spec_fn) in FIXTURES {
        run_fixture(path, &spec_fn());
    }
}

// ---------------------------------------------------------------------------
// JSON-backed fixture tests
// ---------------------------------------------------------------------------
//
// Each `.reference.json` file next to an audio fixture is generated by:
//   python v2/tools/analyze_reference.py ../../fixtures/
//
// The script analyzes the audio with librosa and records raw measurements —
// global and per 5-second segment. This test reads those measurements and
// applies its own tolerance constants (see above) to check that Kaidee
// matches the reference across the whole track, not just at the end.

#[test]
#[ignore = "requires WAV fixtures with .reference.json — see ../../fixtures/README.md"]
fn reference_json_fixtures() {
    let dir = std::path::Path::new("../../fixtures");
    let mut json_files: Vec<_> = std::fs::read_dir(dir)
        .expect("../../fixtures directory missing — add WAV + .reference.json files")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_str().map(|s| s.ends_with(".reference.json")).unwrap_or(false))
        .collect();
    json_files.sort();

    assert!(
        !json_files.is_empty(),
        "No .reference.json files found in ../../fixtures/.\n\
         Run: python v2/tools/analyze_reference.py ../../fixtures/"
    );

    for json_path in &json_files {
        let json_str = json_path.to_str().unwrap();
        // Accept .wav or .mp3 alongside the JSON.
        let stem = json_str.replace(".reference.json", "");
        let audio_path = [format!("{stem}.wav"), format!("{stem}.mp3")]
            .into_iter()
            .find(|p| std::path::Path::new(p).exists())
            .unwrap_or_else(|| panic!("no .wav or .mp3 found for {json_str}"));

        let content = std::fs::read_to_string(json_str)
            .unwrap_or_else(|e| panic!("failed to read {json_str}: {e}"));
        let root: serde_json::Value = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("invalid JSON in {json_str}: {e}"));

        println!("fixture: {}", json_path.file_name().unwrap().to_str().unwrap());

        let (samples, sample_rate) = load_audio_mono(&audio_path)
            .unwrap_or_else(|| panic!("failed to read {audio_path}"));
        let mut pipeline = Pipeline::new(sample_rate as f32, 1);
        let frames = pipeline.push(&samples);
        assert!(!frames.is_empty(), "no frames from {audio_path}");

        // --- Global checks (whole-track tail) ---
        check_global(&root, &frames, &audio_path);

        // --- Per-segment checks (throughout the whole track) ---
        check_temporal(&root["temporal"], &frames, &audio_path);
    }
}

/// Assert global (whole-track tail) values against raw librosa measurements.
fn check_global(root: &serde_json::Value, frames: &[AudioFrame], label: &str) {
    let librosa = &root["per_tool"]["librosa"];
    let tail_secs = 10.0f32;

    // BPM
    if let Some(ref_bpm) = librosa["bpm"].as_f64().map(|v| v as f32) {
        let actual = frames.last().unwrap().bpm;
        assert!(
            (ref_bpm - BPM_TOL..=ref_bpm + BPM_TOL).contains(&actual),
            "{label}: global BPM expected {ref_bpm:.1} ± {BPM_TOL}, got {actual:.2}"
        );
    }

    // Band energies — floor check against librosa's whole-track measurement.
    // dominant_band is not asserted: librosa and Kaidee use different energy
    // aggregation (mean STFT magnitude vs. Euclidean norm of FFT bins) and
    // routinely disagree on which band leads.
    let band_keys = ["sub_bass", "bass", "low_mid", "mid", "presence", "air"];
    let band_fns: &[fn(&AudioFrame) -> f32] = &[
        |f| f.sub_bass,
        |f| f.bass,
        |f| f.low_mid,
        |f| f.mid,
        |f| f.presence,
        |f| f.air,
    ];
    if let Some(band_energies) = librosa.get("band_energies") {
        for (&band, &getter) in band_keys.iter().zip(band_fns) {
            let ref_energy = band_energies[band].as_f64().unwrap_or(0.0) as f32;
            if ref_energy > BAND_ACTIVE_THRESHOLD {
                let min = ref_energy * BAND_ENERGY_FLOOR;
                let v = mean_tail(frames, tail_secs, getter);
                assert!(v >= min, "{label}: global {band} mean {v:.4} < min {min:.4}");
            }
        }
    }

    // RMS is intentionally not asserted here. librosa measures time-domain
    // sample RMS; Kaidee measures spectral RMS from FFT magnitudes. The two
    // operate on different scales and their ratio varies by track, so there
    // is no stable threshold to assert against. RMS correctness is covered
    // by the synthetic unit tests in this file (full_scale_sine_rms_near_one,
    // half_scale_sine_rms_lower_than_full, etc.) where the input is known.
}

/// Assert per-segment values against raw librosa measurements from
/// `temporal` in the reference JSON.
///
/// Kaidee's output frames are sliced into the same time windows as librosa's
/// segments and each window is asserted independently — this catches drift or
/// instability mid-track that a tail-only check would miss.
///
/// BPM is only checked from CONVERGENCE_SECS onward, giving the tracker
/// time to lock before the first assertion.
fn check_temporal(
    temporal: &serde_json::Value,
    frames: &[AudioFrame],
    label: &str,
) {
    // Analysis rate is fixed at 100 Hz (ANALYSIS_RATE_HZ in analysis/mod.rs).
    const ANALYSIS_HZ: f32 = 100.0;

    let segment_secs = temporal
        .get("segment_secs")
        .and_then(|v| v.as_f64())
        .unwrap_or(5.0) as f32;
    let frames_per_seg = (segment_secs * ANALYSIS_HZ) as usize;
    let convergence_seg = (CONVERGENCE_SECS / segment_secs).ceil() as usize;

    let segments = match temporal.get("segments").and_then(|v| v.as_array()) {
        Some(s) => s,
        None => return,
    };

    let band_keys = ["sub_bass", "bass", "low_mid", "mid", "presence", "air"];
    let band_fns: &[fn(&AudioFrame) -> f32] = &[
        |f| f.sub_bass,
        |f| f.bass,
        |f| f.low_mid,
        |f| f.mid,
        |f| f.presence,
        |f| f.air,
    ];

    for seg in segments {
        let idx = seg.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let start_s = seg.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let frame_start = idx * frames_per_seg;
        let frame_end = (frame_start + frames_per_seg).min(frames.len());
        if frame_start >= frames.len() {
            break;
        }
        let seg_frames = &frames[frame_start..frame_end];
        let seg_label = format!("{label} @ {start_s:.0}s");

        // Use the second half of the segment to avoid onset edge effects.
        let half = segment_secs / 2.0;

        // BPM — only after convergence; snapshot on the last frame.
        if idx >= convergence_seg {
            if let Some(ref_bpm) = seg["bpm"].as_f64().map(|v| v as f32) {
                let actual = seg_frames.last().unwrap().bpm;
                assert!(
                    (ref_bpm - BPM_TOL..=ref_bpm + BPM_TOL).contains(&actual),
                    "{seg_label}: BPM expected {ref_bpm:.1} ± {BPM_TOL}, got {actual:.2}"
                );
            }
        }

        // Band energies — floor check against librosa's per-segment measurement.
        if let Some(band_energies) = seg.get("band_energies") {
            for (&band, &getter) in band_keys.iter().zip(band_fns) {
                let ref_energy = band_energies[band].as_f64().unwrap_or(0.0) as f32;
                if ref_energy > BAND_ACTIVE_THRESHOLD {
                    let min = ref_energy * BAND_ENERGY_FLOOR;
                    let v = mean_tail(seg_frames, half, getter);
                    assert!(v >= min, "{seg_label}: {band} mean {v:.4} < min {min:.4}");
                }
            }
        }
    }

    // BPM stability: Kaidee's post-convergence BPM standard deviation (SD)
    // must stay within BPM_STD_MAX_RATIO × librosa's reference SD. This
    // catches mid-track drift that per-segment snapshots might straddle.
    if let Some(ref_std) = temporal["bpm_std"].as_f64().map(|v| v as f32) {
        let max_std = BPM_STD_FLOOR.max(ref_std * BPM_STD_MAX_RATIO);
        let post_start = convergence_seg * frames_per_seg;
        if post_start < frames.len() {
            let bpms: Vec<f32> = frames[post_start..].iter().map(|f| f.bpm).collect();
            let mean = bpms.iter().sum::<f32>() / bpms.len() as f32;
            let variance =
                bpms.iter().map(|b| (b - mean).powi(2)).sum::<f32>() / bpms.len() as f32;
            let std = variance.sqrt();
            assert!(
                std <= max_std,
                "{label}: BPM SD {std:.3} > max {max_std:.3} (ref SD {ref_std:.3} × {BPM_STD_MAX_RATIO})"
            );
        }
    }
}

/// Assert that the band with the highest mean energy over `tail_secs` matches
/// `expected_band`.
fn assert_dominant_band(
    label: &str,
    window: &str,
    frames: &[AudioFrame],
    tail_secs: f32,
    expected_band: &str,
) {
    let means = [
        ("sub_bass", mean_tail(frames, tail_secs, |f| f.sub_bass)),
        ("bass", mean_tail(frames, tail_secs, |f| f.bass)),
        ("low_mid", mean_tail(frames, tail_secs, |f| f.low_mid)),
        ("mid", mean_tail(frames, tail_secs, |f| f.mid)),
        ("presence", mean_tail(frames, tail_secs, |f| f.presence)),
        ("air", mean_tail(frames, tail_secs, |f| f.air)),
    ];
    let dominant = means.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    assert_eq!(
        dominant.0, expected_band,
        "{label} [{window}]: dominant band expected {expected_band}, got {} ({:.3})",
        dominant.0, dominant.1
    );
}

/// Like `run_fixture` but uses a caller-supplied BPM tolerance.
fn run_fixture_with_bpm_tolerance(path: &str, spec: &FixtureSpec, bpm_tol: f32) {
    let (samples, sample_rate) = load_audio_mono(path)
        .unwrap_or_else(|| panic!("failed to read {path}"));
    let mut pipeline = Pipeline::new(sample_rate as f32, 1);
    let frames = pipeline.push(&samples);
    assert!(!frames.is_empty(), "no frames produced from {path}");

    let tail_secs = 10.0f32;

    if let Some(expected_bpm) = spec.bpm {
        let actual = frames.last().unwrap().bpm;
        assert!(
            (expected_bpm - bpm_tol..=expected_bpm + bpm_tol).contains(&actual),
            "{path}: BPM expected {expected_bpm:.1} ± {bpm_tol}, got {actual:.2}"
        );
    }

    // Band energies
    let checks: &[(&str, Option<f32>, fn(&AudioFrame) -> f32)] = &[
        ("sub_bass", spec.sub_bass_min, |f| f.sub_bass),
        ("bass", spec.bass_min, |f| f.bass),
        ("low_mid", spec.low_mid_min, |f| f.low_mid),
        ("mid", spec.mid_min, |f| f.mid),
        ("presence", spec.presence_min, |f| f.presence),
        ("air", spec.air_min, |f| f.air),
    ];
    for &(name, min, getter) in checks {
        if let Some(min) = min {
            let v = mean_tail(&frames, tail_secs, getter);
            assert!(v >= min, "{path}: {name} mean {v:.3} < min {min:.3}");
        }
    }

    if let Some(band) = spec.dominant_band {
        assert_dominant_band(path, "tail", &frames, tail_secs, band);
    }
    if let Some(min) = spec.rms_min {
        let v = mean_tail(&frames, tail_secs, |f| f.rms);
        assert!(v >= min, "{path}: rms mean {v:.3} < min {min:.3}");
    }
    if let Some(max) = spec.rms_max {
        let v = mean_tail(&frames, tail_secs, |f| f.rms);
        assert!(v <= max, "{path}: rms mean {v:.3} > max {max:.3}");
    }
    if let Some(min_peak) = spec.onset_min_peak {
        let n_tail = (tail_secs * 100.0) as usize;
        let tail = if frames.len() > n_tail { &frames[frames.len() - n_tail..] } else { &frames };
        let max_onset = tail.iter().map(|f| f.onset_strength).fold(0.0f32, f32::max);
        assert!(max_onset >= min_peak, "{path}: max onset {max_onset:.3} < {min_peak:.3}");
    }
}

fn run_fixture(path: &str, spec: &FixtureSpec) {
    run_fixture_with_bpm_tolerance(path, spec, spec.effective_bpm_tolerance());
}
