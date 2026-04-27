# Reference Analysis Tools

Tooling for generating and verifying audio fixture references.

The workflow produces `.reference.json` files that encode expected analysis
output (BPM, RMS, band energies) for each audio fixture. These JSON files are
committed to git; the audio files themselves are gitignored.

---

## fixture.sh

The only script you need. Run it from the repository root.

```sh
# Add a new track, generate its reference, verify it passes
./v2/tools/fixture.sh path/to/track.mp3

# Re-analyze all fixtures and run tests (e.g. after an algorithm change)
./v2/tools/fixture.sh

# Regenerate JSON only — skip running tests
./v2/tools/fixture.sh --analyze-only

# Run tests against existing JSON — skip re-analyzing
./v2/tools/fixture.sh --test-only
```

The script sets up its own Python environment on first run — no manual
activation needed.

Fixtures default to `./fixtures` relative to wherever you run the script.
Override with `--fixtures <dir>` if needed.

---

## What the script does

1. **Creates a Python venv** (`v2/tools/.venv`) and installs dependencies from
   `requirements.txt` on first run, or whenever `requirements.txt` changes.

2. **Copies the audio file** into the fixtures directory (if one was given).

3. **Analyzes the audio** with `analyze_reference.py`, which measures BPM, RMS,
   and frequency band energies using two independent tools (librosa + aubio) and
   writes a `.reference.json` file. The JSON records both raw measurements and
   consensus expected values used by the tests.

4. **Runs the fixture tests** (`cargo test … reference_json --include-ignored`),
   which verify that Kaidee's output matches the reference — per segment across
   the whole track, not just at the end.

---

## What is tested

For each 5-second window of the track (independently):
- **BPM** — checked from 10 s onward, once the tracker has converged
- **RMS** range
- **Band energies** — minimum share per active band
- **Dominant band**

Plus over the whole post-convergence region:
- **BPM stability** — standard deviation must stay below `bpm_std_max`

---

## Optional: aubio (second BPM reference)

aubio provides a second independent BPM estimate. When both tools agree the
tolerance stays at ±1 BPM; when they diverge the tolerance widens automatically.

```sh
# macOS
brew install aubio && pip install aubio

# Linux
sudo apt install libaubio-dev && pip install aubio
```

If aubio is not installed the script silently uses librosa alone.

---

## On a new machine

The `.reference.json` files are committed, so fixture tests can run on any
machine without the original audio:

```sh
cargo test --package kaidee-analysis reference_json -- --include-ignored
```

To regenerate the JSON (e.g. you have the audio files and want to re-measure):

```sh
./v2/tools/fixture.sh --analyze-only
```
