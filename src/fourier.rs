//! Fourier transform computation and processing

use crate::math;
use log::info;
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::{collections::VecDeque, sync::Arc};

/// Remove DC offset before computing a Fourier transform
const REMOVE_DC: bool = true;

/// Fast and sane approximation of a constant-Q transform
///
/// The constant-Q transform is a cousin of the Fourier transform whose bins are
/// distributed exponentially, rather than linearly. This better matches human
/// perception, which is roughly logarithmic in frequency, but unfortunately
/// this transform also has two problems:
///
/// - The FFT trick does not trivially apply to the constant-Q transform.
/// - A short-term constant-Q transform has a an input length that either
///   diverges to infinity at low frequencies (which is intractable) or
///   converges to zero at high frequencies (which is useless).
///
/// We address the first problem by approximating the constant-Q transform as
/// a weighted average of radix-2 FFTs, and the second problem by bounding
/// the set of radix-2 FFTs that we will use for STFT to a useful amount.
///
// FIXME: This currently computes obviously wrong results (no activity in bins
//        on the right), figure out why.
//
pub struct SteadyQTransform {
    /// Radix-2 FFTs used to approximate the constant-Q transform, and frequency
    /// bin of the base (first) FFT on which each one is considered optimal.
    ffts_and_optimal_bins: Box<[(FourierTransform, f32)]>,

    /// Weights to be used when transitioning from one radix-2 FFT to the next
    transition_weights: Box<[Box<[f32]>]>,

    /// Buffer to merge all the FFT outputs into one
    merged_output: Box<[Complex<f32>]>,
}
//
impl SteadyQTransform {
    /// Get ready to compute approximate constant-Q transforms with a certain
    /// frequency resolution at 20Hz (in Hz) and time resolution at 20kHz
    /// (in ms), given the audio sampling rate and a choice of window function.
    pub fn new(
        freq_res_at_20hz: f32,
        time_res_at_20khz: f32,
        sample_rate: usize,
        window: &str,
    ) -> Self {
        // Translate the low-frequency resolution into a first FFT length
        let mut fft_len_at_20hz = FourierTransform::fft_len(freq_res_at_20hz, sample_rate);
        let inv_bin_width_at_20hz = FourierTransform::inv_bin_width(fft_len_at_20hz, sample_rate);

        // Translate the high-frequency time resolution into a last FFT length
        let samples_at_20khz = (time_res_at_20khz * sample_rate as f32 / 1000.0) as usize;
        let fft_len_at_20khz = if samples_at_20khz.is_power_of_two() {
            samples_at_20khz
        } else {
            (samples_at_20khz / 4).next_power_of_two()
        };
        info!(
            "At a sampling rate of {} Hz, achieving a time resolution of {} ms requires a {}-points FFT",
            sample_rate,
            time_res_at_20khz,
            fft_len_at_20khz
        );

        // If the time resolution constraint is harsher than the frequency
        // resolution one, pick the FFT length accordingly.
        if fft_len_at_20khz > fft_len_at_20hz {
            info!(
                "Can achieve desired time-frequency resolution compromise with a single {}-points FFT",
                fft_len_at_20khz
            );
            fft_len_at_20hz = fft_len_at_20khz;
        }

        // Check that the constant-Q transform can fulfill those constraints
        // There is a factor of 1000 between the start and the end of the range,
        // so we cannot cover that range with more than 11 FFTs (base FFT +
        // decimations 1/2, 1/4, 1/8, ..., 1/1024.
        debug_assert!(fft_len_at_20hz.is_power_of_two());
        let fft_len_at_20hz_pow2 = fft_len_at_20hz.trailing_zeros();
        let fft_len_at_20khz_pow2 = fft_len_at_20khz.trailing_zeros();
        let num_ffts = (fft_len_at_20hz_pow2 - fft_len_at_20khz_pow2 + 1) as usize;
        assert!(
            num_ffts <= 11,
            "Cannot achieve requested time-frequency resolution compromise ({} Hz at 20Hz, {} ms at 20kHz)",
            freq_res_at_20hz, time_res_at_20khz
        );

        // Set up all the radix-2 FFTs required to approximate a constant-Q
        // transform, and record on which bin of the 20Hz FFT we consider each
        // of these radix-2 FFTs to be an optimal approximation. Spread the FFTs
        // around the center of the 20Hz-20kHz log scale.
        let mut planner = RealFftPlanner::<f32>::new();
        let mut ffts_and_optimal_bins = VecDeque::new();
        let center_freq = (20.0f32 * 20_000.0).sqrt() * inv_bin_width_at_20hz;
        let center_right_len = 2usize.pow((fft_len_at_20hz_pow2 + fft_len_at_20khz_pow2) / 2);
        let mut pick_fft = |freq, len| {
            info!(
                "Will use a {}-points FFT at {} Hz",
                len,
                freq / inv_bin_width_at_20hz
            );
            (
                FourierTransform::from_fft(planner.plan_fft_forward(len), window),
                freq,
            )
        };
        let (mut left_freq, mut left_len, mut right_freq, mut right_len);
        if num_ffts % 2 == 0 {
            left_freq = center_freq / std::f32::consts::SQRT_2;
            left_len = center_right_len * 2;
            right_freq = center_freq * std::f32::consts::SQRT_2;
            right_len = center_right_len;
        } else {
            ffts_and_optimal_bins.push_front(pick_fft(center_freq, center_right_len));
            left_freq = center_freq / 2.0;
            left_len = center_right_len * 2;
            right_freq = center_freq * 2.0;
            right_len = center_right_len / 2;
        }
        while ffts_and_optimal_bins.len() < num_ffts {
            ffts_and_optimal_bins.push_front(pick_fft(left_freq, left_len));
            left_freq /= 2.0;
            left_len *= 2;
            ffts_and_optimal_bins.push_back(pick_fft(right_freq, right_len));
            right_freq *= 2.0;
            right_len /= 2;
        }
        debug_assert_eq!(ffts_and_optimal_bins.len(), num_ffts);
        let ffts_and_optimal_bins: Box<[_]> = ffts_and_optimal_bins.drain(..).collect();
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
        self.first_fft_mut().input()
    }

    /// Query the output length
    pub fn output_len(&self) -> usize {
        self.first_fft().output_len()
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

        // Compute the first FFT (this will garble its input, so do it last)
        first_fft.window_and_compute_fft();

        // For the lowest frequencies, follow the first (widest) FFT
        let low_bins = first_optimal_bin.ceil() as usize;
        self.merged_output[..low_bins].copy_from_slice(&first_fft.output[..low_bins]);

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

        // For the highest frequencies, follow interpolant of the last (narrowest) FFT
        let (last_fft, last_optimal_bin) = self.ffts_and_optimal_bins.last().unwrap();
        let high_bins = last_optimal_bin.ceil() as usize;
        let last_fft_interpolant = math::interpolate_c32(
            &last_fft.output[..],
            2usize.pow(self.ffts_and_optimal_bins.len() as u32 - 1),
        );
        for (dest, src) in self
            .merged_output
            .iter_mut()
            .zip(last_fft_interpolant)
            .skip(high_bins)
        {
            *dest = src
        }

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

/// Short-term Fourier transform
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
    #[allow(unused)]
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
    #[allow(unused)]
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
