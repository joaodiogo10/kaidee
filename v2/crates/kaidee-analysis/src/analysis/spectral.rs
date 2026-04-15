//! Spectral analysis — short-time FFT (fast Fourier transform) with a Hann
//! window, producing a magnitude spectrum and per-band energies.

use super::to_01;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub struct Spectral {
    fft: Arc<dyn Fft<f32>>,
    scratch: Vec<Complex<f32>>,
    hann: Vec<f32>,
    fft_buf: Vec<Complex<f32>>,
    magnitudes: Vec<f32>,
    // Circular sample buffer — always holds the most recent FFT_SIZE samples.
    // New samples overwrite the oldest; write_pos is where the next write goes.
    window: Vec<f32>,
    write_pos: usize,
    sample_rate: f32,
}

pub struct BandEnergies {
    pub sub_bass: f32,
    pub bass: f32,
    pub low_mid: f32,
    pub mid: f32,
    pub presence: f32,
    pub air: f32,
}

impl Spectral {
    pub const FFT_SIZE: usize = 2048;

    pub fn new(sample_rate: f32) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(Self::FFT_SIZE);
        let scratch = vec![Complex::new(0.0f32, 0.0); fft.get_inplace_scratch_len()];

        // Hann window — tapers the edges of each FFT frame to zero. Without
        // this, sharp edges at frame boundaries create false high-frequency
        // content.
        let hann: Vec<f32> = (0..Self::FFT_SIZE)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (Self::FFT_SIZE as f32 - 1.0)).cos())
            })
            .collect();

        Self {
            fft,
            scratch,
            hann,
            fft_buf: vec![Complex::new(0.0f32, 0.0); Self::FFT_SIZE],
            magnitudes: vec![0.0f32; Self::FFT_SIZE / 2],
            window: vec![0.0f32; Self::FFT_SIZE],
            write_pos: 0,
            sample_rate,
        }
    }

    /// Push one mono sample into the circular buffer.
    pub fn push_sample(&mut self, sample: f32) {
        self.window[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % Self::FFT_SIZE;
    }

    /// Number of positive-frequency FFT bins — half of `FFT_SIZE`.
    /// Used by downstream processors (e.g. the beat tracker) that need to
    /// allocate state proportional to the spectrum size.
    pub fn n_bins(&self) -> usize {
        Self::FFT_SIZE / 2
    }

    /// Run the FFT on the current sample window and fill the internal magnitude
    /// buffer. Call `magnitudes()` or `band_energies()` afterwards.
    pub fn process(&mut self) {
        for i in 0..Self::FFT_SIZE {
            let idx = (self.write_pos + i) % Self::FFT_SIZE;
            self.fft_buf[i] = Complex::new(self.window[idx] * self.hann[i], 0.0);
        }

        self.fft
            .process_with_scratch(&mut self.fft_buf, &mut self.scratch);

        // Only the first half of FFT output is useful (positive frequencies).
        // The second half mirrors it (Nyquist symmetry for real input).
        // Normalize by FFT_SIZE so magnitudes don't scale with window size.
        let n_bins = Self::FFT_SIZE / 2;
        for (i, c) in self.fft_buf[..n_bins].iter().enumerate() {
            self.magnitudes[i] = c.norm() * 2.0 / Self::FFT_SIZE as f32;
        }
    }

    pub fn magnitudes(&self) -> &[f32] {
        &self.magnitudes
    }

    pub fn band_energies(&self) -> BandEnergies {
        let hz_per_bin = (self.sample_rate / 2.0) / self.magnitudes.len() as f32;

        // Total RMS (root mean square) energy within a frequency band
        // [lo_hz, hi_hz). Sum of squared magnitudes (not mean) so wide bands
        // (MID = 71 bins, AIR = 744 bins) get proportional credit relative to
        // narrow bands (SUB = 2 bins, BASS = 9 bins).
        let band_rms = |lo_hz: f32, hi_hz: f32| -> f32 {
            let lo = (lo_hz / hz_per_bin) as usize;
            let hi = ((hi_hz / hz_per_bin) as usize + 1).min(self.magnitudes.len());
            if lo >= hi {
                return 0.0;
            }
            let total_sq: f32 = self.magnitudes[lo..hi].iter().map(|m| m * m).sum();
            total_sq.sqrt()
        };

        BandEnergies {
            sub_bass: to_01(band_rms(20.0, 60.0)),
            bass: to_01(band_rms(60.0, 250.0)),
            low_mid: to_01(band_rms(250.0, 500.0)),
            mid: to_01(band_rms(500.0, 2000.0)),
            presence: to_01(band_rms(2000.0, 6000.0)),
            air: to_01(band_rms(6000.0, self.sample_rate / 2.0)),
        }
    }
}
