# Analysis Module

The analysis layer turns raw interleaved audio samples into [`AudioFrame`]
snapshots at a fixed rate. Components are composed by [`Pipeline`]
(see [`pipeline.rs`](pipeline.rs)) and expose their results through
[`AudioFrame`] (see [`frame.rs`](../frame.rs)).

## Global parameters

Parameters shared across all components. Each symbol below has its own
anchor so component docs can link to it directly rather than duplicate
the value. Everything else is local to a component and derived from its
own base constants.

### `analysis_rate`

**100 Hz.** Defined in [`global.rs`](../global.rs) as `ANALYSIS_RATE_HZ`.

Frame production rate. All per-frame derivations (hop size, ring-buffer
capacities, PLL phase increment, …) scale from this.

Component docs link to this symbol as
`[analysis_rate](../analysis.md#analysis_rate)`.

## Shared types

| Type               | Defined in                           | Purpose                                                                                                     |
| ------------------ | ------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `AudioFrame`       | [`frame.rs`](../frame.rs)            | One analysis-tick snapshot. The public API boundary — consumers (UI, renderer) read these, never raw audio. |
| `BandEnergies`     | [`spectral/mod.rs`](spectral/mod.rs) | Six-band energy split (sub_bass → air), emitted by every `SpectralAnalyzer`.                                |
| `SpectralAnalyzer` | [`spectral/mod.rs`](spectral/mod.rs) | Swappable spectral-analysis strategy. Pipeline accepts any implementor.                                     |

## Components

| Component  | Doc                                    | Summary                                                                                                                             |
| ---------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `beat`     | [`beat/beat.md`](beat/beat.md)         | BPM estimation (pair-product comb + prior) driving a PLL phase tracker.                                                             |
| `spectral` | [`spectral/plan.md`](spectral/plan.md) | Strategy interface + implementations (FFT, IIR filterbank). Sibling docs: [`fft.md`](spectral/fft.md), [`iir.md`](spectral/iir.md). |
| `dynamics` | [`dynamics.md`](dynamics.md)           | RMS and peak tracking.                                                                                                              |
| `levels`   | —                                      | Internal helpers (`to_01`, range mapping).                                                                                          |
| `pipeline` | —                                      | Glue layer: owns the active spectral strategy and beat tracker, produces one `AudioFrame` per `1 / analysis_rate` seconds of audio. |

Each component doc owns its own base constants and derived-quantities
tables. Cross-component references link to the symbols defined here
(e.g. [`analysis_rate`](#analysis_rate)) or to the component's own
exported symbols — never a duplicated numeric value.
