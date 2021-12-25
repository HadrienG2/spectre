//! General-purpose math utilities

mod simd;

use realfft::num_complex::Complex;

pub use simd::sum_f32;

/// Interpolate a table of complex numbers into a series that is ~Nx larger
pub fn interpolate_c32(
    input: &[Complex<f32>],
    stride: usize,
) -> impl Iterator<Item = Complex<f32>> + '_ {
    assert!(stride > 0);
    let inv_stride = 1.0 / stride as f32;
    input
        .windows(2)
        .flat_map(move |pair| {
            let left = pair[0];
            let right = pair[1];
            (0..stride).map(move |idx| {
                let weight = idx as f32 * inv_stride;
                (1.0 - weight) * left + weight * right
            })
        })
        .chain(input.last().cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn interpolate_c32(input: Vec<f32>, stride: usize) -> TestResult {
        // Ignore silly configurations and keep output below 4KB
        if input.iter().any(|x| !x.is_finite())
            || stride == 0
            || input.len().saturating_mul(stride) > 1_024
        {
            return TestResult::discard();
        }

        // Turn real input into complex input
        let input = input
            .windows(2)
            .map(|pair| Complex::new(pair[0], pair[1]))
            .collect::<Box<[_]>>();

        // Perform the interpolation
        let output = super::interpolate_c32(&input[..], stride).collect::<Box<[_]>>();

        // Check that the interpolant has the right length
        let expected_len = if input.len() > 0 {
            stride * (input.len() - 1) + 1
        } else {
            0
        };
        assert_eq!(output.len(), expected_len);

        // Check that the interpolant has the right values
        for (idx, &output) in output.iter().enumerate() {
            let left_idx = idx / stride;
            let right_idx = (left_idx + 1).min(input.len() - 1);
            let left = input[left_idx];
            let right = input[right_idx];
            let weight = (idx % stride) as f32 / stride as f32;
            assert_eq!(output, (1.0 - weight) * left + weight * right);
        }
        TestResult::passed()
    }
}
