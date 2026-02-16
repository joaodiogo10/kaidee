# Kaidee

Audio-reactive visual engine for live DJ sets.

> **Disclaimer:** This project is 100% AI-generated, including all source code, tests, and documentation.

Kaidee turns any audio source into real-time visuals. It listens to music — from a file, microphone, or system audio loopback — and generates reactive imagery by analyzing bass, mids, highs, and beat patterns. Images and source code files are pre-compiled into optimized visual assets (abstract, pixel-sorted, displaced, solarized, etc.), then rendered live through six visual modes with a drift system that evolves parameters over time. Supports GPU-accelerated rendering via OpenGL, MIDI controller mapping, keyboard-driven parameter control, and live recording to MP4.

https://github.com/user-attachments/assets/demo.mp4

## Install

```bash
pip install -e ".[all]"
```

Or install only what you need:

```bash
pip install -e "."                # core only (compile)
pip install -e ".[live]"          # + live renderer (sounddevice, moderngl)
pip install -e ".[midi]"          # + MIDI support
```

## Workflow

```
1. Put images in input/images/ and source code in input/source/
2. Compile assets:  kaidee-compile
3. Run live:        kaidee-live --play track.mp3
```

### Compile

Pre-processes images and source code into optimized visual assets. Run this once before a live set, or whenever you change your input files.

```bash
# Default: images from input/images/, source from input/source/
kaidee-compile

# Custom source paths (files, directories, or globs)
kaidee-compile --source ~/projects/myapp src/utils.py

# Skip image compilation (source code only)
kaidee-compile --no-images --source .

# Custom output directory
kaidee-compile --output my_compiled/
```

This generates a `compiled/` directory with `.npy` frames and a `meta.json` manifest. Each input image produces ~20 variants (abstract, solarized, color-rotated, posterized, pixel-sorted, radial/fluid displaced). Source code files are rendered as syntax-highlighted images in multiple styles (terminal, dense, scatter, blueprint, neon).

### Live

```bash
# Play a file with synced visuals
kaidee-live --play track.mp3

# Live mic input
kaidee-live --device 0

# System audio loopback (requires BlackHole or similar)
kaidee-live --loopback

# Demo mode (no audio hardware needed)
kaidee-live --demo

# With MIDI controller
kaidee-live --play track.mp3 --midi

# List available devices
kaidee-live --list-devices
kaidee-live --list-midi
```

### CLI Flags

| Flag             | Description                              |
| ---------------- | ---------------------------------------- |
| `--play FILE`    | Play audio file and sync visuals         |
| `--device N`     | Audio input device index                 |
| `--loopback`     | Capture system audio via loopback device |
| `--demo`         | Synthetic 128 BPM audio (no hardware)    |
| `--mode MODE`    | Starting visual mode (default: `auto`)   |
| `--fps N`        | Target framerate (default: `60`)         |
| `--midi [NAME]`  | Enable MIDI controller                   |
| `--list-devices` | List audio devices and exit              |
| `--list-midi`    | List MIDI devices and exit               |

## Visual Modes

| #   | Mode           | Description                  |
| --- | -------------- | ---------------------------- |
| 1   | `feedback`     | Recursive feedback loops     |
| 2   | `glitch`       | Digital glitch artifacts     |
| 3   | `radial`       | Radial/circular patterns     |
| 4   | `kaleidoscope` | Mirrored kaleidoscope        |
| 5   | `strobe`       | Strobe flash effects         |
| 6   | `displacement` | Pixel displacement mapping   |
| 7   | `auto`         | Auto-selects based on energy |

## Keyboard Controls

### General

| Key       | Action                         |
| --------- | ------------------------------ |
| `Esc`     | Quit                           |
| `F`       | Toggle fullscreen              |
| `H`       | Toggle HUD overlay             |
| `Space`   | Pause/resume (file playback)   |
| `Up/Down` | Volume up/down                 |
| `-/+`     | Decrease/increase target FPS   |
| `Q/E`     | Decrease/increase render scale |
| `D`       | Switch to demo audio           |
| `G`       | Toggle drift system on/off     |
| `F9`      | Start/stop recording           |

### Mode Selection

| Key   | Action             |
| ----- | ------------------ |
| `1-7` | Select visual mode |

### Parameters

| Key   | Parameter    | Range     |
| ----- | ------------ | --------- |
| `Z/X` | Reactivity   | 0.0 - 1.0 |
| `,/.` | Perception   | 0.0 - 1.0 |
| `A/S` | Movement     | 0.0 - 1.0 |
| `B/N` | Color        | 0.0 - 1.0 |
| `J/K` | Brightness   | 0.0 - 1.0 |
| `C/V` | Blend        | 0.0 - 1.0 |
| `</>` | Code overlay | 0.0 - 1.0 |
| `U/I` | Image blend  | 0.0 - 1.0 |
| `[/]` | Zoom         | 0.0 - 1.0 |
| `T/Y` | Pan X        | 0.0 - 1.0 |
| `O/P` | Pan Y        | 0.0 - 1.0 |

### Randomize

| Key | Action                                                   |
| --- | -------------------------------------------------------- |
| `W` | Cycle wildness level (calm → mild → wild → wilder → any) |
| `R` | Randomize parameters at current wildness level           |

### Auto-Randomize

| Key | Action                                                          |
| --- | --------------------------------------------------------------- |
| `0` | Toggle auto-randomize on/off                                    |
| `9` | Cycle auto-randomize interval (1, 2, 4, 8, 16, 32, random bars) |

Auto-randomize triggers on bar boundaries using the current wildness level.

### Presets

| Key        | Action                            |
| ---------- | --------------------------------- |
| `Ctrl+1-9` | Save current state to preset slot |
| `Alt+1-9`  | Load preset from slot             |

### MIDI

| Key                   | Action                         |
| --------------------- | ------------------------------ |
| `M`                   | Toggle MIDI learn mode         |
| `1-7` (in learn mode) | Assign last moved knob to mode |

## Credits

- Example 1: Tatlo — _Groovy Tool EP_ — 03 Groovy Tool 3

## GPU Rendering

Kaidee requires OpenGL 3.3+ for rendering. The `moderngl` package (included in the `[live]` extra) handles all post-processing on the GPU: zoom, pan, rotation, chromatic aberration, color grading, vignette, and trail persistence.

## System Audio Loopback

To visualize system audio (Spotify, browser, DJ software), install a virtual audio device:

- **macOS**: [BlackHole](https://existential.audio/blackhole/) (free)
- **Windows**: [VB-Cable](https://vb-audio.com/Cable/)
- **Linux**: PulseAudio Monitor (built-in)

Then run `kaidee-live --loopback` to auto-detect the loopback device.
