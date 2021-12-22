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

    /// Scratch space
    scratch: Box<[Complex<f32>]>,

    /// Complex FFT output
    output: Box<[Complex<f32>]>,

    /// Complex FFT magnitude in dB
    magnitude: Box<[f32]>,
}
//
impl FourierTransform {
    /// Get ready to compute Fourier transforms with a certain frequency resolution (in Hz)
    pub fn new(resolution: f32, sample_rate: usize) -> Self {
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
        let fft_len = 2_usize.pow((sample_rate as f32 / resolution).log2().ceil() as _);
        info!(
            "At a sampling rate of {} Hz, achieving the requested frequency resolution of {} Hz requires a {}-points FFT",
            sample_rate,
            resolution,
            fft_len
        );

        // Prepare for the FFT computation
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_len);
        let input = fft.make_input_vec().into_boxed_slice();
        let scratch = fft.make_scratch_vec().into_boxed_slice();
        let output = fft.make_output_vec().into_boxed_slice();
        let magnitude = vec![0.0; output.len()].into_boxed_slice();

        // Return the state to the client
        Self {
            fft,
            input,
            scratch,
            output,
            magnitude,
        }
    }

    /// Access the input buffer
    pub fn input(&mut self) -> &mut [f32] {
        &mut self.input[..]
    }

    /// Compute the Fourier transform and return coefficient magnitudes in dBFS
    pub fn compute(&mut self) -> &[f32] {
        // Remove DC offset if configured to do so
        if REMOVE_DC {
            let average = simd::sum_f32(&self.input[..]) / self.input.len() as f32;
            self.input.iter_mut().for_each(|elem| *elem -= average);
        }

        // Compute FFT
        self.fft
            .process_with_scratch(
                &mut self.input[..],
                &mut self.output[..],
                &mut self.scratch[..],
            )
            .expect("Failed to compute FFT");

        // Normalize amplitudes, convert to dBFS, and send the result out
        let norm_sqr = 1.0 / self.input.len() as f32;
        for (coeff, mag) in self.output.iter().zip(self.magnitude.iter_mut()) {
            // NOTE: dBFS formula is 20*log10(|coeff| / sqrt(N)) but we avoid a
            //       bunch of square roots by noticing that by definition of the
            //       logarithm this is equal to 10*log10(|coeff|² / N).
            *mag = 10.0 * (coeff.norm_sqr() * norm_sqr).log10();
        }
        &self.magnitude[..]
    }
}
