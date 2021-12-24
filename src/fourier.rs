//! Fourier transform computation and processing

use crate::math;
use log::info;
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Remove DC offset before computing a Fourier transform
const REMOVE_DC: bool = true;

/// Fast approximation of a constant-Q transform
///
/// The constant-Q transform is a cousin of the Fourier transform whose bins are
/// distributed exponentially, rather than linearly, which better matches human
/// perception. Unfortunately, this transform does not enjoy the luxury of a
/// super-fast algorithm like its linearly spaced cousin, so we approximate it
/// as a weighted average of "easy" FFTs.
///
// FIXME: This currently computes obviously wrong results (no activity in bins
//        on the right), figure out why.
//
//        I suspect the basic premise of dividing FFT length by 2 indefinitely
//        until 20kHz is reached is also flawed and should be rethought. An
//        alternative would be to specify a frequency resolution at 20Hz and
//        a response time at 20kHz, deduce a number of decimations from that
//        and use those decimations. If so, must make sure to fill the end of
//        merged_output with data from the last FFT.
//
//        Another path to think about is to be constant-Q in the middle of the
//        log frequency scale (which is where most interesting things happen)
//        and standard FFT on the sides. After all, specifying a frequency
//        resolution at 20Hz and a time resolution at 20kHz does not specify
//        where the transition between these two constraints should happen.
//
pub struct ApproxConstantQTransform {
    /// Radix-2 FFTs used to approximate the constant-Q transform, and frequency
    /// bin of the base (first) FFT on which each one is considered optimal.
    ffts_and_optimal_bins: Box<[(FourierTransform, f32)]>,

    /// Weights to be used when transitioning from one radix-2 FFT to the next
    transition_weights: Box<[Box<[f32]>]>,

    /// Buffer to merge all the FFT outputs into one
    merged_output: Box<[Complex<f32>]>,
}
//
impl ApproxConstantQTransform {
    /// Get ready to compute approximate constant-Q transforms with a certain
    /// frequency resolution at 20Hz (in Hz), given the audio sampling rate and
    /// a choice of window function.
    pub fn new(resolution_at_20hz: f32, sample_rate: usize, window: &str) -> Self {
        // Translate the desired frequency resolution into an FFT length
        let fft_len_at_20hz = FourierTransform::fft_len(resolution_at_20hz, sample_rate);
        let inv_bin_width_at_20hz = FourierTransform::inv_bin_width(fft_len_at_20hz, sample_rate);

        // Set up all the radix-2 FFTs required to approximate a constant-Q
        // transform, and record on which bin of the 20Hz FFT we consider each
        // of these radix-2 FFTs to be an optimal approximation
        let mut planner = RealFftPlanner::<f32>::new();
        let mut optimal_freq = 10.0;
        let mut fft_len = 2 * fft_len_at_20hz;
        let mut ffts_and_optimal_bins = Vec::new();
        while optimal_freq < 20_000.0 {
            fft_len = (fft_len / 2).max(2);
            optimal_freq *= 2.0;
            ffts_and_optimal_bins.push((
                FourierTransform::from_fft(planner.plan_fft_forward(fft_len), window),
                optimal_freq * inv_bin_width_at_20hz,
            ));
        }
        let ffts_and_optimal_bins = ffts_and_optimal_bins.into_boxed_slice();
        let merged_output = ffts_and_optimal_bins[0].0.output.clone();

        // For each consecutive pair of radix-2 FFTs, determine the weights to
        // use so that the transition from one to the next is smooth when the
        // transform is rendered on a log frequency scale.
        let transition_weights = ffts_and_optimal_bins
            .windows(2)
            .map(|pair| {
                let (_fft1, bin1) = &pair[0];
                let (_fft2, bin2) = &pair[1];
                let start_idx = bin1.ceil() as usize;
                let end_idx = bin2.ceil() as usize;
                (start_idx..end_idx)
                    // FIXME: Cross-check this formula
                    .map(|idx| ((idx as f32).log2() - bin1.log2()) / (bin2.log2() - bin1.log2()))
                    .collect()
            })
            .collect();

        // Return the resulting constant-Q FFT approximation harness
        Self {
            ffts_and_optimal_bins,
            transition_weights,
            merged_output,
        }
    }

    /// Access the input buffer
    pub fn input(&mut self) -> &mut [f32] {
        &mut self.first_fft_mut().input[..]
    }

    /// Query the output length
    pub fn output_len(&self) -> usize {
        self.first_fft().output.len()
    }

    /// Compute the constant-Q transform approximation and return coefficient
    /// magnitudes in dBFS.
    pub fn compute(&mut self) -> &[f32] {
        // Prepare the first FFT's input
        let (first_fft, other_ffts) = self.ffts_and_optimal_bins.split_at_mut(1);
        let (ref mut first_fft, first_optimal_bin) = first_fft[0];
        first_fft.prepare_input();

        // Propagate the end of that input to other FFTs'inputs and compute them
        let first_input = first_fft.input();
        for (fft, _optimal_bin) in other_ffts.iter_mut() {
            let input = fft.input();
            input.copy_from_slice(&first_input[first_input.len() - input.len()..]);
            fft.window_and_compute_fft();
        }

        // Compute the first FFT (this will garble its input, so we do it last)
        first_fft.window_and_compute_fft();

        // For inaudible frequencies below 20Hz, follow the first (widest) FFT
        let bins_below_20hz = first_optimal_bin.ceil() as usize;
        self.merged_output[..bins_below_20hz].copy_from_slice(&first_fft.output[..bins_below_20hz]);

        // After that, combine pairs of consecutive radix-2 FFTs using the
        // previously determined weights. Bear in mind that those FFTs must be
        // interpolated in order to match the frequency resolution of the
        // final merged FFT.
        for (idx, (fft_pair, transition_weights)) in self
            .ffts_and_optimal_bins
            .windows(2)
            .zip(self.transition_weights.iter())
            .enumerate()
        {
            // Extract the pair of FFTs that we're going to work with
            let (ref fft1, optimal_bin1) = fft_pair[0];
            let (ref fft2, optimal_bin2) = fft_pair[1];

            // Determine the target bin index range in the merged FFT
            let start_idx = optimal_bin1.ceil() as usize;
            let end_idx = optimal_bin2.ceil() as usize;
            debug_assert_eq!(end_idx - start_idx, transition_weights.len());

            // Determine how bins of each FFT map into bins of the merged FFT
            let stride1 = 2usize.pow(idx as u32);
            let stride2 = 2 * stride1;

            // Produce linear interpolants of each FFT on the merged FFT's bins
            let fft1_interpolant = math::interpolate_c32(&fft1.output[..], stride1);
            let fft2_interpolant = math::interpolate_c32(&fft2.output[..], stride2);

            // Perform the FFT merging
            for ((dest, (src1, src2)), weight) in self
                .merged_output
                .iter_mut()
                .zip(fft1_interpolant.zip(fft2_interpolant))
                .take(start_idx)
                .zip(transition_weights.iter())
            {
                *dest = (1.0 - weight) * src1 + weight * src2;
            }
        }

        // TODO: After max frequency, only use interpolant of the narrowest FFT

        // Compute the magnitude of the merged FFT
        FourierTransform::compute_magnitudes(
            &self.merged_output[..],
            &mut self.ffts_and_optimal_bins[0].0.magnitude[..],
        )
    }

    // Access the first (widest) inner FFT
    fn first_fft(&self) -> &FourierTransform {
        &self.ffts_and_optimal_bins[0].0
    }
    //
    fn first_fft_mut(&mut self) -> &mut FourierTransform {
        &mut self.ffts_and_optimal_bins[0].0
    }
}

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
        Self::compute_magnitudes(&self.output[..], &mut self.magnitude[..])
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
        let output_norm = 2.0 / math::sum_f32(&window[..]);
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
            let average = math::sum_f32(&self.input[..]) / self.input.len() as f32;
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
    fn compute_magnitudes<'mag>(
        output: &[Complex<f32>],
        magnitude: &'mag mut [f32],
    ) -> &'mag [f32] {
        // Normalize magnitudes, convert to dBFS, and send the result out
        for (coeff, mag) in output.iter().zip(magnitude.iter_mut()) {
            // NOTE: dBFS formula is 20*log10(|coeff|) but we avoid a
            //       bunch of square roots by noticing that by definition of the
            //       logarithm this is equal to 10*log10(|coeff|²).
            // TODO: Right now, log10 computations are about 30% of the CPU
            //       consumption. This could be sped up, at the expense of
            //       losing precision, by using the floating-point exponent as
            //       an (integral) approximation of the log2. But that's only
            //       3dB precision, which is very low. Maybe a bit of iterative
            //       refinement could get us to 0.something at low-ish cost.
            *mag = 10.0 * (coeff.norm_sqr()).log10();
        }
        magnitude
    }
}
