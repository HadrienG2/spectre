//! Fourier transform resampling for desired display width

use crate::simd;

// Integrate the linear interpolant of a tabulated function between two
// fractional bin coordinates.
fn integrate(f: &[f32], start: f32, end: f32) -> f32 {
    // Find the integral bin range surrounded by start and end
    debug_assert!(end >= start);
    debug_assert!(start >= 0.0);
    debug_assert!(end <= (f.len() - 1) as f32);
    let after_start = start.ceil() as usize;
    let before_end = end.floor() as usize;

    // Compute the value of the interpolant at the start of the range
    let start_fract = start.fract();
    let left_val = if start_fract == 0.0 {
        f[after_start]
    } else {
        (1.0 - start_fract) * f[after_start - 1] + start_fract * f[after_start]
    };
    let end_fract = end.fract();
    let right_val = if end_fract == 0.0 {
        f[before_end]
    } else {
        (1.0 - end_fract) * f[before_end] + end_fract * f[before_end + 1]
    };

    // Dispatch on the right logic depending on this range's characteristics
    if before_end < after_start {
        // Integrating in the middle of a single input bin
        debug_assert_eq!(before_end, after_start - 1);
        0.5 * (left_val + right_val) * (end - start)
    } else {
        // Integrating across at least one bin boundary
        //
        // Contribution before the first bin boundary
        let left_average = 0.5 * (left_val + f[after_start]);
        let left_width = 1.0 - start_fract;
        let left_contrib = left_average * left_width;
        //
        // Contribution after the last bin boundary
        let right_average = 0.5 * (f[before_end] + right_val);
        let right_width = end_fract;
        let right_contrib = right_average * right_width;
        //
        // Contribution from the bins surrounded by (start, end)
        let middle_contrib = if before_end > after_start {
            // Integrating across at least one integer bin
            0.5 * (f[after_start] + f[before_end]) + simd::sum_f32(&f[after_start + 1..before_end])
        } else {
            // Integrating across one bin border only
            0.0
        };
        //
        // Total integral
        left_contrib + right_contrib + middle_contrib
    }
}

/// Fourier transform resampler
///
/// Converts the native Fourier transform into a format that is suitable for
/// display. Each display bin is modeled as representing a certain frequency
/// range, and its value is the average of a linear FFT interpolant across
/// this frequency range.
///
pub struct FourierResampler {
    /// Output bin borders
    bin_borders: Box<[f32]>,

    /// Output bin averaging weights (= reverse bin width)
    bin_weights: Box<[f32]>,

    /// Resampled FFT storage
    output_bins: Box<[f32]>,
}
//
impl FourierResampler {
    /// Prepare for Fourier transform resampling
    pub fn new(
        transform_len: usize,
        sample_rate: usize,
        num_output_bins: usize,
        min_freq: f32,
        max_freq: f32,
        log_scale: bool,
    ) -> Self {
        // Compute the Fourier transform bin width and deduce the fractional bin
        // position corresponding to the minimum and maximum frequency.
        assert!(transform_len >= 2);
        assert!(sample_rate > 0);
        assert!(num_output_bins >= 1 && num_output_bins < i32::MAX as usize);
        assert!(min_freq >= 0.0);
        assert!(max_freq > min_freq);
        assert!(max_freq <= (sample_rate / 2) as f32);
        let bin_width = (sample_rate / 2) as f32 / (transform_len - 1) as f32;
        let min_bin = min_freq / bin_width;
        let max_bin = max_freq / bin_width;

        // Find the list of bin borders corresponding to the resampled transform
        let bin_borders: Box<[_]> = if log_scale {
            (0..=num_output_bins as i32)
                .map(|b| min_bin * (max_bin / min_bin).powf(b as f32 / num_output_bins as f32))
                .collect()
        } else {
            (0..=num_output_bins as i32)
                .map(|b| min_bin + b as f32 * (max_bin - min_bin) / num_output_bins as f32)
                .collect()
        };

        // Compute the averaging weights
        let bin_weights = bin_borders
            .windows(2)
            .map(|w| 1.0 / (w[1] - w[0]))
            .collect();

        // Return the resulting resamplign harness
        Self {
            bin_borders,
            bin_weights,
            output_bins: vec![0.0; num_output_bins].into_boxed_slice(),
        }
    }

    /// Resample a Fourier transform
    pub fn resample(&mut self, fourier: &[f32]) -> &[f32] {
        for (bin, (borders, &weight)) in self
            .output_bins
            .iter_mut()
            .zip(self.bin_borders.windows(2).zip(&self.bin_weights[..]))
        {
            *bin = integrate(fourier, borders[0], borders[1]) * weight;
        }
        &self.output_bins[..]
    }
}
