//! Fourier transform computation and processing

use crate::simd;
use log::info;
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Remove DC offset before computing a Fourier transform
const REMOVE_DC: bool = true;

/// Fourier transform processor
pub struct FourierTransform {
    /// FFT implementation
    fft: Arc<dyn RealToComplex<f32>>,

    /// Time series input
    input: Box<[f32]>,

    /// Window to be applied to input data
    window: Box<[f32]>,

    /// Scratch space
    scratch: Box<[Complex<f32>]>,

    /// Complex FFT output
    output: Box<[Complex<f32>]>,

    /// Complex FFT magnitude in dB
    magnitude: Box<[f32]>,
}
//
impl FourierTransform {
    /// Get ready to compute Fourier transforms with a certain frequency
    /// resolution (in Hz), given the audio sample rate and a choice of
    /// window function.
    pub fn new(resolution: f32, sample_rate: usize, window: &str) -> Self {
        let fft_len = Self::fft_len(resolution, sample_rate);
        let mut planner = RealFftPlanner::<f32>::new();
        Self::from_fft(planner.plan_fft_forward(fft_len), window)
    }

    /// Access the input buffer
    pub fn input(&mut self) -> &mut [f32] {
        &mut self.input[..]
    }

    /// Query the output length
    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    /// Compute the Fourier transform and return coefficient magnitudes in dBFS
    pub fn compute(&mut self) -> &[f32] {
        self.prepare_input();
        self.window_and_compute_fft();
        self.compute_magnitudes(&self.output[..])
    }

    /// Determine the right FFT length to reach a certain frequency resolution,
    /// knowing the underlying audio sampling rate
    fn fft_len(resolution: f32, sample_rate: usize) -> usize {
        // Translate the desired frequency resolution into an FFT length
        //
        // Given 2xN input data point, a real-fft produces N+1 frequency bins
        // ranging from 0 frequency to sampling_rate/2. So bins spacing df is
        // sampling_rate/(2*N) Hz.
        //
        // By inverting this relation, we get that the smallest N needed to achieve
        // a bin spacing smaller than df is Nmin = sampling_rate / (2 * df). We turn
        // back that Nmin to a number of points 2xNmin, and we round that to the
        // next power of two.
        //
        assert!(resolution > 0.0);
        assert_ne!(sample_rate, 0);
        let fft_len = 2_usize.pow((sample_rate as f32 / resolution).log2().ceil() as _);
        info!(
            "At a sampling rate of {} Hz, achieving the requested frequency resolution of {} Hz requires a {}-points FFT",
            sample_rate,
            resolution,
            fft_len
        );
        fft_len
    }

    /// Knowing an FFT length and the underlying audio sampling rate, deduce the
    /// inverse of the FFT bin width.
    fn inv_bin_width(fft_len: usize, sample_rate: usize) -> f32 {
        let output_len = fft_len / 2;
        let max_freq = sample_rate / 2;
        output_len as f32 / max_freq as f32
    }

    /// Subset of the constructor that happens after an FFT has been planned
    fn from_fft(fft: Arc<dyn RealToComplex<f32>>, window: &str) -> Self {
        // Prepare for the FFT computation
        let input = fft.make_input_vec().into_boxed_slice();
        let scratch = fft.make_scratch_vec().into_boxed_slice();
        let output = fft.make_output_vec().into_boxed_slice();
        let magnitude = vec![0.0; output.len()].into_boxed_slice();

        // Prepare for input windowing
        let mut window: Box<[_]> = match window {
            "rectangular" => std::iter::repeat(1.0).take(input.len()).collect(),
            "triangular" => (0..input.len() / 2)
                .chain((0..input.len() / 2).rev())
                .map(|x| x as f32 / ((input.len() - 1) / 2) as f32)
                .collect(),
            "hann" => (0..input.len())
                .map(|n| {
                    (std::f32::consts::PI * n as f32 / (input.len() - 1) as f32)
                        .sin()
                        .powi(2)
                })
                .collect(),
            "blackman" => (0..input.len())
                .map(|n| {
                    use std::f32::consts::TAU;
                    let alpha = 0.16;
                    let a0 = 0.5 * (1.0 - alpha);
                    let a1 = 0.5;
                    let a2 = 0.5 * alpha;
                    let phase = TAU * n as f32 / input.len() as f32;
                    a0 - a1 * (phase).cos() + a2 * (2.0 * phase).cos()
                })
                .collect(),
            "nuttall" => (0..input.len())
                .map(|n| {
                    use std::f32::consts::TAU;
                    let a0 = 0.355768;
                    let a1 = 0.487396;
                    let a2 = 0.144232;
                    let a3 = 0.012604;
                    let phase = TAU * n as f32 / input.len() as f32;
                    a0 - a1 * (phase).cos() + a2 * (2.0 * phase).cos() - a3 * (3.0 * phase).cos()
                })
                .collect(),
            _ => panic!("Window type {} is not supported", window),
        };

        // Pre-normalize the window function so that output is normalized
        let output_norm = 2.0 / simd::sum_f32(&window[..]);
        for x in window.iter_mut() {
            *x *= output_norm;
        }

        // Return the state to the client
        Self {
            fft,
            input,
            window,
            scratch,
            output,
            magnitude,
        }
    }

    /// Prepare the input data for the FFT computation
    fn prepare_input(&mut self) {
        // Remove DC offset if configured to do so
        if REMOVE_DC {
            let average = simd::sum_f32(&self.input[..]) / self.input.len() as f32;
            self.input.iter_mut().for_each(|elem| *elem -= average);
        }
    }

    /// Window the input data and compute the FFT
    fn window_and_compute_fft(&mut self) {
        // Apply window function
        for (x, &w) in self.input.iter_mut().zip(self.window.iter()) {
            *x *= w;
        }

        // Compute FFT
        self.fft
            .process_with_scratch(
                &mut self.input[..],
                &mut self.output[..],
                &mut self.scratch[..],
            )
            .expect("Failed to compute FFT");
    }

    /// Compute FFT magnitudes in dBFS and return them
    fn compute_magnitudes(&mut self, output: &[Complex<f32>]) -> &[f32] {
        // Normalize magnitudes, convert to dBFS, and send the result out
        for (coeff, mag) in output.iter().zip(self.magnitude.iter_mut()) {
            // NOTE: dBFS formula is 20*log10(|coeff|) but we avoid a
            //       bunch of square roots by noticing that by definition of the
            //       logarithm this is equal to 10*log10(|coeff|²).
            // TODO: Right now, log10 computations are about 30% of the CPU
            //       consumption. This could be sped up, at the expense of
            //       losing precision, by using the floating-point exponent as
            //       an (integral) approximation of the log2. Precision can be
            //       improved by dividing by 2^N and applying this procedure
            //       recursively.
            *mag = 10.0 * (coeff.norm_sqr()).log10();
        }
        &self.magnitude[..]
    }
}
