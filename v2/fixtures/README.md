# Audio fixtures

Reference audio files and their `.reference.json` measurements live here.

Audio files (`.wav`, `.mp3`) are **gitignored** — too large for the repository.
The `.reference.json` files **are tracked** — they encode the expected analysis
output and let fixture tests run on any machine without the audio.

## Adding a fixture

```sh
# From the repository root
./v2/tools/fixture.sh path/to/track.mp3
```

This copies the file here, generates the `.reference.json`, and runs the tests.
See [v2/tools/README.md](../tools/README.md) for full usage.

## Supported formats

- **WAV** — 16-bit or 32-bit PCM, or 32-bit float
- **MP3** — CBR (constant bit rate) or VBR (variable bit rate)

Stereo is mixed down to mono automatically. Any sample rate is accepted.
Minimum duration: 30 seconds.

## Reproducing on a new machine

The `.reference.json` files are committed, so tests run without the audio:

```sh
cargo test --package kaidee-analysis reference_json -- --include-ignored
```

To regenerate JSON after obtaining the audio files:

```sh
./v2/tools/fixture.sh --analyze-only
```
