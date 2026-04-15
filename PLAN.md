# Kaidee v2 — Project Plan

## Why v2

v1 proved the concept works. The problems that warrant a full rewrite:

- **Performance ceiling ~30 fps** — Python + NumPy CPU compositing (~24ms/frame) cannot hit 60fps reliably. The GIL prevents real parallelism.
- **Beat sync is imprecise** — BPM detection via FFT autocorrelation + consensus voting has high latency (~1s update cycle). Beat phase drifts under tempo changes.
- **Analysis and rendering are tightly coupled** — v1 mixes audio analysis, parameter logic, and rendering in ways that make each hard to improve independently.

## Goals

1. Solid 60fps at 1080p
2. Musical beat sync (sub-10ms beat phase error)
3. **Audio analysis as a first-class foundation** — the visual engine is a consumer of the analysis layer, not entangled with it
4. Clean crate boundaries so each layer can be tested and developed independently

---

## Tech Stack

| Concern | Choice | Reason |
|---|---|---|
| Language | **Rust** | No GIL, zero-cost abstractions, true parallelism |
| Audio I/O | `cpal` | Cross-platform, low-latency callbacks |
| FFT | `rustfft` | Fast, no-alloc in hot path |
| GPU | `wgpu` | WebGPU API, cross-platform (Metal/Vulkan/DX12) |
| Windowing | `winit` | Pairs with wgpu, event loop |
| Image loading | `image` | Standard Rust image crate |
| Serialization | `serde` + `serde_json` | Config, presets, meta.json |

---

## Architecture

### Crate Structure

```
kaidee/
├── crates/
│   ├── kaidee-analysis/    # lib — pure audio analysis, zero UI/rendering code
│   ├── kaidee-renderer/    # lib — GPU visual engine, zero audio code (Phase 2)
│   └── kaidee-app/         # bin — composition root, owns the event loop
├── v1/                     # Original Python implementation (archived)
└── PLAN.md
```

**Dependency rule:** `kaidee-analysis` and `kaidee-renderer` never depend on each other.
Only `kaidee-app` depends on both.

### Visualizer Trait

The UI/display layer is abstracted behind a trait in `kaidee-app`. This is what makes
the terminal → GPU swap possible without touching any analysis code:

```rust
// kaidee-app
pub trait Visualizer {
    fn update(&mut self, frame: &AudioFrame);
    fn render(&mut self);
}

// Phase 1 — terminal bars
struct TerminalUi { ... }
impl Visualizer for TerminalUi { ... }

// Phase 2 — GPU renderer
struct GpuRenderer { ... }
impl Visualizer for GpuRenderer { ... }
```

`kaidee-app` holds a `Box<dyn Visualizer>` and never knows which one it is.

### Data Flow

```
cpal audio callback (low-latency thread)
        │
        ▼
  RingBuffer<f32>  ←── lock-free, never blocks callback
        │
        ▼
  Analysis thread  (dedicated OS thread, lives in kaidee-analysis)
        │
        ├── FFT pipeline
        ├── Band energies
        ├── PLL beat tracker
        ├── Onset detector
        ├── Chroma / key
        ├── Dynamics (LUFS, RMS)
        ├── Spatial (stereo width, M/S)
        └── Structure (build/drop/breakdown)
        │
        ▼
  AudioFrame  ──→  published via triple-buffer (no blocking on render thread)
        │
        ▼
  Event bus  (derived higher-level musical events)
     DROP | BUILD | BREAKDOWN | PHRASE_START | KEY_CHANGE
        │
        ▼
  kaidee-app: Box<dyn Visualizer>::update(frame)
        │
        ├── Phase 1: TerminalUi  (prints band bars to stdout)
        └── Phase 2: GpuRenderer (wgpu, 60Hz, modes, post-FX)
```

The `AudioFrame` is the public API boundary. The visualizer never touches raw audio or FFT output.

---

## AudioFrame — Core Data Model

```rust
pub struct AudioFrame {
    pub timestamp: Duration,

    // Spectral
    pub spectrum: Vec<f32>,         // full FFT magnitudes (linear scale)
    pub bands: BandEnergies,        // sub-bass, bass, low-mid, mid, presence, air
    pub spectral_centroid: f32,     // perceived brightness (Hz)
    pub spectral_flux: f32,         // rate of spectral change
    pub spectral_flatness: f32,     // 0=tonal, 1=noise-like

    // Rhythm
    pub bpm: f32,
    pub beat_phase: f32,            // 0–1 position within current beat
    pub bar_phase: f32,             // 0–1 position within current bar
    pub beat_confidence: f32,       // PLL confidence score
    pub onset_strength: f32,

    // Harmonic
    pub chroma: [f32; 12],          // pitch class profile (C, C#, D, ...)
    pub key: Option<MusicalKey>,
    pub harmonic_energy: f32,       // energy in harmonic partials
    pub percussive_energy: f32,     // energy in transients

    // Dynamics
    pub rms: f32,
    pub lufs_short: f32,            // ~400ms window perceptual loudness
    pub peak: f32,
    pub crest_factor: f32,          // transient density
    pub dynamic_range: f32,

    // Spatial
    pub stereo_width: f32,          // 0=mono, 1=full stereo
    pub mid_energy: f32,
    pub side_energy: f32,
    pub phase_correlation: f32,     // 1=in phase, -1=out of phase

    // Timbre
    pub mfcc: [f32; 13],            // mel-frequency cepstral coefficients
    pub warmth: f32,                // low-mid / high ratio
    pub roughness: f32,             // zero crossing rate proxy

    // Structure
    pub music_intensity: f32,       // slow EMA of energy
    pub drop_impact: f32,           // impulse on detected drops, decays ~0.7s
    pub structure_phase: StructurePhase, // Intro | Build | Peak | Breakdown | Drop
}
```

---

## Analysis Layers

### 1. Spectral

- Hanning window → `rustfft` → magnitude spectrum
- Mel filterbank (128 bins) for perceptual band energies
- Band boundaries (Hz): sub-bass <60, bass 60–250, low-mid 250–500, mid 500–2k, presence 2k–6k, air >6k
- Spectral centroid, flux, flatness computed from FFT output

### 2. Rhythm — PLL Beat Tracker

Replace v1's autocorrelation voting with a Phase-Locked Loop:

```
onset envelope → tempo estimator (autocorrelation, updated ~1/sec)
              ↓
         PLL phase tracker
              │  error = onset_strength × sin(2π × phase_error)
              │  phase += (bpm/60) × dt + k_p × error
              ↓
         beat_phase, bar_phase, beat_pulse (sharp on-beat transient)
```

PLL gives sub-10ms beat phase error and tracks tempo changes smoothly. Beat confidence from phase error magnitude.

### 3. Harmonic

- Harmonic-percussive source separation (HPSS) via median filtering on spectrogram
- Chromagram from harmonic component (12 pitch classes)
- Key detection: Krumhansl-Schmuckler key profiles vs chroma vector

### 4. Dynamics & Loudness

- RMS over 50ms window
- Short-term LUFS (ITU-R BS.1770-4, K-weighting filter + 400ms integration)
- True peak (4× oversampled)
- Crest factor = peak / RMS

### 5. Spatial

- Mid = (L + R) / 2, Side = (L - R) / 2
- Stereo width = RMS(side) / RMS(mid)
- Phase correlation = normalized cross-correlation of L/R at lag 0

### 6. Timbre

- 13 MFCCs from mel spectrogram (standard librosa convention)
- Zero crossing rate as roughness proxy
- Warmth = low-mid energy / (presence + air energy)

### 7. Structure Detection

Stateful classifier watching slow trends:
- `Breakdown`: energy_slow < 0.3 and dropping
- `Build`: onset density increasing over 8+ bars
- `Drop`: energy_jump > 0.15 AND bass > 0.35 AND intensity_delta > 0.05
- `Peak`: sustained high energy (> 0.7 for > 4 bars)

---

## Visual Engine (Phase 2)

Port v1's 6 modes and GPU pipeline, now as Rust traits:

```rust
pub trait Mode: Send {
    fn render(&mut self, frame: &AudioFrame, assets: &Assets, params: &DerivedParams) -> RgbImage;
}
```

GPU post-processing shader (WGSL, WebGPU) ports directly from v1's GLSL. Same 13 effects, same uniform layout.

---

## Milestones

### Phase 1 — Analysis Foundation
- [ ] Repo setup, crate skeleton
- [ ] `cpal` audio capture → lock-free ring buffer
- [ ] FFT pipeline + band energies
- [ ] `AudioFrame` struct (spectral + dynamics)
- [ ] PLL beat tracker
- [ ] Onset detector
- [ ] Harmonic analysis (HPSS + chroma)
- [ ] Spatial analysis
- [ ] Timbre (MFCCs)
- [ ] Structure classifier
- [ ] Validation tool: real-time spectrum + beat phase display (terminal or simple wgpu window)

### Phase 2 — Visual Engine
- [ ] `wgpu` window setup
- [ ] Image compilation pipeline (Rust, replaces Python `compile.py`)
- [ ] Asset loader (`.npz` → GPU textures or native format)
- [ ] Port post-FX fragment shader to WGSL
- [ ] FBO ping-pong (trail persistence)
- [ ] Port 6 modes to Rust traits
- [ ] Auto-mode pool + crossfade

### Phase 3 — Application
- [ ] Parameter system (11 knobs, derived params)
- [ ] Drift system (port from v1)
- [ ] Keyboard input
- [ ] MIDI (auto-map)
- [ ] Preset save/load
- [ ] Recording (ffmpeg pipe)
- [ ] File audio playback + seeking
- [ ] HUD overlay

---

## v1 Reference

The original Python implementation is preserved in [v1/](v1/). Architecture documented in [v1/ARCHITECTURE.md](v1/ARCHITECTURE.md).
