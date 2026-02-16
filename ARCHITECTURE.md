# Architecture

Technical reference for the Kaidee audio-reactive visual engine.

## Overview

Kaidee has two phases: an offline **compile** step that pre-processes images and source code into optimized visual assets, and a **live** renderer that analyzes audio in real time and drives GPU-accelerated visuals. The live renderer runs at 60 Hz, composing frames on the CPU (mode rendering, image blending, code overlay) then uploading to the GPU for post-processing (zoom, pan, chromatic aberration, color grading, trails, vignette).

**~4,200 lines of Python** | NumPy, SciPy, Pillow, Pygame, ModernGL, librosa, sounddevice

## Project Structure

```
kaidee3/
├── pyproject.toml          # Package config, entry points, extras
├── compile.py              # Wrapper → src.compile:main
├── live.py                 # Wrapper → src.live:main
├── presets.json            # Saved parameter presets (slots 1-9)
├── compiled/               # Pre-compiled .npz frames + meta.json
├── input/
│   ├── images/             # Source images for visual generation
│   └── source/             # Source code files for code visualization
├── recordings/             # MP4 output from F9 recording
├── src/
│   ├── live.py             # LiveRenderer + main loop
│   ├── compile.py          # Asset compilation pipeline
│   ├── audio.py            # AudioAnalyzer, FileAudio, DemoAudio
│   ├── gpu.py              # OpenGL 3.3 post-processing pipeline
│   ├── assets.py           # Compiled asset loading + RAM cache
│   ├── params.py           # Parameter system + derived values
│   ├── drift.py            # Temporal evolution system
│   ├── postprocess.py      # Post-FX uniform computation
│   ├── transforms.py       # Image transforms (blend, zoom, solarize, etc.)
│   ├── hud.py              # HUD text overlay
│   ├── input.py            # Keyboard event handling + randomize
│   ├── midi.py             # MIDI controller auto-mapping
│   ├── recorder.py         # Background MP4 encoding via ffmpeg
│   └── modes/
│       ├── base.py         # Abstract Mode base class
│       ├── feedback.py     # Recursive zoom tunnel
│       ├── glitch.py       # Digital glitch artifacts
│       ├── radial.py       # Concentric pulsating waves
│       ├── kaleidoscope.py # Mirrored mandala patterns
│       ├── strobe.py       # Stroboscopic hard-switching
│       ├── displacement.py # Fluid pixel warping
│       ├── helpers.py      # Pan helpers, color rotation lookup
│       └── __init__.py     # Mode registry + auto-mode pool
└── tests/
```

## Data Flow

```
COMPILE (offline, run once):

  input/images/*.jpg  ──┐
                        ├──→  20 variants per image  ──→  compiled/*.npz + meta.json
  input/source/*.*   ──┘     5 code render styles

LIVE (real-time, 60 Hz):

  Audio Source (file / mic / demo)
       │
       ▼
  FFT Analysis ──→ bass, mids, highs, energy, onset
       │              │
       │              ▼
       │         Drift System ──→ DriftValues (intensity, chaos, drop, beat phase, ...)
       │              │
       ▼              ▼
  Mode.render(audio, drift, assets) ──→ uint8 RGB frame
       │
       ├── Image superimposition (blend layers)
       ├── Code overlay + displacement
       ├── Ghost image overlay
       │
       ▼
  GPU Post-Processing (fragment shader)
       │   zoom, pan, rotation, chromatic aberration,
       │   saturation, contrast, hue shift, brightness,
       │   trail persistence, vignette, flash, chaos
       │
       ▼
  Display (pygame.display.flip)
       │
       └── Recording (optional, background ffmpeg pipe)
```

## Audio Analysis

Three interchangeable audio sources share the same interface (bass, mids, highs, energy, onset, BPM, beat/bar tracking).

### AudioAnalyzer (real-time mic/loopback)

- **Callback-based** via sounddevice — processes ~46ms blocks (44.1 kHz, 2048 samples)
- **FFT pipeline:** Hanning window → `np.fft.rfft` → frequency binning:
  - Bass: < 300 Hz
  - Mids: 300–4000 Hz
  - Highs: >= 4000 Hz
  - Energy: `sqrt(mean(audio^2))`
  - Onset: `max(0, new_energy - prev_energy) * 10`
- **BPM detection** (background thread, ~1/sec):
  - 8-second ring buffer → decimate 16:1 → block energy → onset envelope
  - FFT autocorrelation of onset → peak detection with octave folding
  - Consensus voting over last 8 estimates
- Optional raw audio buffering for recording

### FileAudio (pre-analyzed file playback)

- Loads audio at 22.05 kHz mono via librosa
- Pre-computes mel spectrogram (128 bins, hop=512):
  - Bass: mel bins 0–15
  - Mids: mel bins 16–79
  - Highs: mel bins 80–127
- Beat tracking via `librosa.beat.beat_track()`
- Playback via `pygame.mixer`
- CDJ-style seeking with acceleration (Left/Right arrow keys)

### DemoAudio (synthetic, no hardware)

- 128 BPM baseline with simulated kick/hihat/mid patterns
- No I/O overhead — useful for development and testing

## Parameter System

### Params (11 knobs, all 0.0–1.0)

| Parameter | Default | Controls |
|-----------|---------|----------|
| reactivity | 0.5 | Bass/energy response intensity |
| perception | 0.5 | Overlap, chaos, trail amount |
| movement | 0.5 | Rotation, pan, zoom speed |
| color | 0.5 | Chromatic aberration, hue shift |
| brightness | 0.5 | Exponential brightness (3^(2x-1)) |
| blend | 0.5 | Secondary mode alpha |
| img_blend | 0.5 | Image crossfade + layering |
| code | 0.5 | Code overlay visibility |
| zoom | 0.0 | Manual zoom boost |
| pan_x, pan_y | 0.5 | Manual pan offset |

### DerivedParams (computed from Params)

Params are mapped through power curves to produce usable ranges:

```
reactivity  → react_mod = 0.2 + reactivity² × 2.8       (range: 0.2–3.0)
perception  → warp, overlap, trail, chaos  (via p^2.5)
movement    → rotation_speed, zoom_amount, drift_speed  (via m^1.5)
color       → chroma, color_intensity, color_shift_speed  (via c^2.5)
img_blend   → img_crossfade, img_layers (1–6), img_layer_alpha
```

### Presets

- 9 slots saved to `presets.json`
- Each slot stores all 11 Params + active mode
- Save: Ctrl+1-9 | Load: Alt+1-9

## Drift System

The drift system provides slow-evolving modulation so visuals change over time even with static audio.

### DriftState (rolling accumulators)

- `ema_energy_slow` / `ema_energy_fast` — exponential moving averages (~30s / ~3s windows)
- `ema_bass_slow` — bass EMA for drop detection
- `onset_timestamps` — last 10s of onset events
- `energy_min_recent` — rolling minimum (~8s window)
- `drop_decay` — impulse on drops, decays ~0.7s

### DriftValues (per-frame output)

| Field | Description |
|-------|-------------|
| intensity | Time arc + music intensity boost |
| warmth | Periodic sine (120s period / BPM-modulated) |
| code_mix | Base code visibility + beat pulse modulation |
| chaos | Time arc × 0.4 + music × 0.4 + drop × 0.4 |
| vignette | Tightens on breakdowns, opens on drops |
| mode_energy | Music intensity × 0.5 + time_arc × 0.3 |
| time_arc | Piecewise curve across set (0.3 → 0.8 → 1.0 → 0.7) |
| music_intensity | ema_energy_slow × 0.6 + ema_bass × 0.4 |
| drop_impact | 1.0 on detected drop, exponential decay |
| beat_phase | 0–1 position within current beat |
| bar_phase | 0–1 position within current bar |
| beat_pulse | (1 - beat_phase × 4)² — sharp transient on beat |
| bar_pulse | (1 - bar_phase × 2)² — wider pulse on bar start |

### Drop Detection

```python
energy_jump = ema_fast - energy_min_recent
if energy_jump > 0.15 and intensity_delta > 0.05 and bass > 0.35:
    drop_decay = min(1.0, drop_decay + 0.6)   # impulse
else:
    drop_decay *= max(0, 1.0 - dt * 1.5)      # decay ~0.7s
```

### Time Arc (piecewise over set duration)

```
  0–10%  → 0.3 → 0.5  (warm-up)
 10–40%  → 0.5 → 0.8  (build)
 40–70%  → 0.8 → 1.0  (peak)
 70–85%  → 1.0 → 0.7  (cooldown)
 85–100% → 0.7 → 1.0  (finale)
```

## Visual Modes

All modes implement the same `render()` interface and return a `(H, W, 3)` uint8 RGB frame.

### Auto-Mode Pool

Auto mode selects based on `drift.mode_energy` and cycles every 4 bars:

| Energy | Pool |
|--------|------|
| < 0.35 (calm) | feedback, radial, displacement |
| 0.35–0.65 (mid) | feedback, radial, kaleidoscope, displacement |
| >= 0.65 (intense) | glitch, strobe, kaleidoscope, feedback |

Crossfades during the last beat of each 4-bar phrase.

### Mode Details

**FeedbackMode** — Recursive zoom tunnel. Maintains a persistent float32 buffer that gets zoomed each frame (positive feedback loop). Bass controls zoom intensity, onset triggers channel swaps. Decay prevents overflow. Color source injected on beats.

**GlitchMode** — Digital artifacts. 3-way blend of abstract/sorted/inverted images. Band displacement (4–8 horizontal bands with sinusoidal shifts). Scanning inversion bar. RGB channel separation driven by highs. Solarize overlay on strong bass.

**RadialMode** — Concentric waves. Crossfades between three radial displacement variants (amplitude 30/60/90). Breathing zoom with beat-synced punch. Accumulating rotation driven by mids. Deep abstract blend during low energy.

**KaleidoscopeMode** — Mandala symmetry. Pre-computed polar coordinate grid. Maps pixels to kaleidoscope wedges via modulo + mirror. Segment count: `max(3, min(12, 4 + highs × 4 + bass × 2))`. Bass controls wobble amplitude.

**StrobeMode** — Hard switching. Beat-synced variant selection (inverted/solarized/posterized). Zoom punch on beats. Additive flash on onset. Persistence trail blending. Scanning line darkening on highs.

**DisplacementMode** — Fluid warping. Crossfades between three fluid displacement variants (amplitude 40/80/120). Sinusoidal wave distortion modulated by mids/energy/beat_pulse. Pan drift with audio-modulated phase.

### Complement Pairs (for secondary blend)

```python
COMPLEMENTS = {
    "feedback": "kaleidoscope",
    "glitch": "feedback",
    "radial": "displacement",
    "kaleidoscope": "radial",
    "strobe": "glitch",
    "displacement": "strobe",
}
```

## GPU Pipeline

OpenGL 3.3 core profile via ModernGL. All post-processing runs in a single fragment shader.

### Textures

| Texture | Format | Resolution | Purpose |
|---------|--------|------------|---------|
| tex_frame | RGB8 | render res | CPU-composed frame |
| tex_secondary | RGB8 | render res | Complement mode frame |
| tex_trail | RGB8 | display res | FBO ping-pong (persistence) |
| tex_vignette | R32F | render res | Precomputed radial mask |
| tex_hud | RGBA8 | screen res | HUD text overlay |

### Fragment Shader Effects (applied in order)

1. **Zoom** — `(coord - center) / zoom + center` with wrapping
2. **Pan + rotation shift** — Direct coordinate offset
3. **Chromatic aberration** — Separate R/G/B channel UV offsets
4. **Secondary blend** — Lerp with complement mode texture
5. **Trail persistence** — Lerp with previous FBO (ping-pong)
6. **Saturation** — Grayscale mix
7. **Contrast** — Amplify deviation from 0.5
8. **Hue shift** — Additive RGB offsets
9. **Beat flash** — Additive white
10. **Chaos noise** — Per-pixel hash noise
11. **Chaos scanlines** — Horizontal line darkening
12. **Brightness** — Multiplicative scale
13. **Vignette** — Multiply by radial mask

### Uniforms (14 values per frame)

```
u_zoom, u_pan, u_rot_shift, u_saturation, u_contrast,
u_hue_shift, u_brightness, u_chroma, u_chroma_v,
u_flash, u_chaos, u_chaos_spacing, u_blend_alpha,
u_trail_keep, u_vig_strength, u_has_secondary, u_has_trail, u_time
```

### FBO Ping-Pong

Two framebuffer objects alternate each frame. One renders the new frame while the other is sampled for trail persistence (`u_trail_keep` controls blend factor).

### GPU Skip System

To reduce CPU load, only every Nth frame runs the full CPU composition pipeline. Intermediate frames re-render the same GPU texture with updated uniforms (audio-reactive post-FX still update). The skip count auto-tunes based on actual FPS.

## Compilation Pipeline

`kaidee-compile` pre-processes input materials into optimized `.npz` frames.

### Image Compilation

Each input image produces ~20 variants:

| Variant | Transform |
|---------|-----------|
| sharp | Original (resized to target resolution) |
| abstract | Heavy Gaussian blur (sigma=50) + saturation boost |
| sorted | Pixel sort (rows sorted by brightness) |
| deep | Multi-pass abstract (3× blur + solarize + color rotate) |
| radial_30/60/90 | Sinusoidal radial displacement (3 amplitudes) |
| fluid_40/80/120 | Turbulent wave distortion (3 amplitudes) |

On-the-fly variants derived at load time (not stored):
- inverted, solarized (3 thresholds), color-rotated (5 angles), posterized (3 levels)

### Code Compilation

Source code files are analyzed (entropy, bracket density, nesting depth, token density) and rendered as images in 5 styles:

| Style | Description |
|-------|-------------|
| terminal | Large token-colored text on dark background |
| dense | Small characters at full density |
| scatter | Randomly placed code chunks |
| blueprint | Cyan text on blueprint grid |
| neon | Large neon-colored fragments on black |

### Output

- `compiled/*.npz` — compressed NumPy arrays (uint8 RGB)
- `compiled/meta.json` — manifest with resolution, frame keys, code features

## Post-FX Computation

`compute_postfx()` translates audio + drift + params into the 14 GPU uniforms.

### Drop Flavors

Six visual styles cycle on `bar_idx % 6`, applied when `drop_impact > 0`:

| Flavor | Character |
|--------|-----------|
| Full impact | Saturation + contrast + flash + chroma + zoom + brightness |
| Flash + bright | Strong flash, brightness boost |
| Color burst | High saturation, extreme chroma |
| Zoom only | Dramatic zoom punch |
| Contrast + darken | Deep contrast with brightness dip |
| Chroma + saturation | Color channel separation |

## Recording

The `Recorder` class pipes raw RGB frames to an ffmpeg subprocess on a background thread.

1. Spawn ffmpeg: rawvideo stdin → libx264 (CRF 28, fast preset)
2. Queue frames from render thread (deque, no blocking)
3. On stop: drain queue, close pipe, ffmpeg exits
4. Duration correction via `-itsscale` (wall clock / encoded duration)
5. Audio muxing: sliced from file (FileAudio) or recorded WAV (AudioAnalyzer)
6. Output: `recordings/rec_YYYYMMDD_HHMMSS.mp4`

## Performance Architecture

At 1920x1080 display with 75% render scale (1440x810):

- **CPU composition:** Mode rendering + image blending + code overlay (~24ms per frame)
- **GPU post-processing:** Shader execution + texture upload (~15ms, cached secondary every 4th frame)
- **HUD + display flip:** ~8ms (includes OpenGL buffer swap)
- **GPU skip:** Auto-tunes 1–4 (only every Nth frame runs CPU composition; others re-render with updated uniforms)

### Key Optimizations

- Render at 75% scale, GPU upscales to display resolution
- Secondary mode texture cached (re-dispatched every 4th CPU frame)
- Per-frame `_img_blend` cache avoids duplicate crossfade computation
- HUD dirty-flag caching + font cache (only rebuilds when state changes)
- `blend()` uses uint16 integer math (avoids float32 conversion)
- Glitch band count capped at 8, small shifts skipped
- Background BPM thread (doesn't block audio callback)
- Async recording (background thread + queue)

## MIDI

Auto-maps the first 7 unique CC messages to parameters in order. First 7 unique notes map to modes 1–7. Manual learn mode (press M) allows reassignment.
