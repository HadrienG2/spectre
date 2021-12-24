//! General-purpose math utilities

mod simd;

use realfft::num_complex::Complex;

pub use simd::sum_f32;

/// Interpolate a table of complex numbers into a series that is ~Nx larger
pub fn interpolate_c32(
    input: &[Complex<f32>],
    stride: usize,
) -> impl Iterator<Item = Complex<f32>> + '_ {
    assert!(input.len() > 0);
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
        .chain(std::iter::once(input.last().unwrap().clone()))
}

/// 1% precise log10 approximation for inputs in [0, 1] range
pub fn log10_1pct(x: f32) -> f32 {
    // Handle strange and small inputs
    debug_assert!(!x.is_nan() && x >= 0.0 && x <= 1.0);
    if x.is_subnormal() || x == 0.0 {
        return -f32::INFINITY;
    }

    // Extract the floating-point exponent using IEEE-754 sorcery
    assert_eq!(f32::RADIX, 2);
    const F32_BITS: u32 = std::mem::size_of::<f32>() as u32 * 8;
    const MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
    const EXPONENT_BITS: u32 = F32_BITS - MANTISSA_BITS - 1;
    let exponent_bits = (x.to_bits() >> MANTISSA_BITS) & ((1 << EXPONENT_BITS) - 1);
    let exponent = exponent_bits as i32 - 2i32.pow(EXPONENT_BITS - 1) + 1;

    // The exponent is the integer part of x.log2(). We normalize x by that,
    // which gives us a number w in the [0.5; 2[ range whose log2 is the
    // fractional part of x.log2().
    let w = x / 2.0f32.powi(exponent);

    // We use the Taylor expansion of log2 around 1 to approximate w.log2()
    let w_m1 = w - 1.0;
    use std::f32::consts::LN_2;
    let coeffs = [
        1.0 / LN_2,
        -1.0 / (2.0 * LN_2),
        1.0 / (3.0 * LN_2),
        -1.0 / (4.0 * LN_2),
    ];
    let mut polynomial = [
        w_m1,
        w_m1 * w_m1,
        (w_m1 * w_m1) * w_m1,
        (w_m1 * w_m1) * (w_m1 * w_m1),
    ];
    for (monome, &coeff) in polynomial.iter_mut().zip(coeffs.iter()) {
        *monome *= coeff;
    }
    let log2_w = polynomial.iter().rev().sum::<f32>();

    // From this, we trivially deduce an approximation of x.log2(), that we can
    // turn into an approximation of x.log10().
    let log2_x = exponent as f32 + log2_w;
    1.0 / std::f32::consts::LOG2_10 * log2_x
}

#[cfg(test)]
mod tests {
    use super::*;
    use more_asserts::*;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn interpolate_c32(input: Vec<f32>, stride: usize) -> TestResult {
        // Ignore silly configurations and keep output below 4KB
        if input.len() < 2
            || input.iter().any(|x| !x.is_finite())
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
        assert_eq!(output.len(), stride * (input.len() - 1) + 1);

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

    #[quickcheck]
    fn log10_1pct(x: f32) -> TestResult {
        // Ignore incompatible inputs
        if x.is_nan() || x < 0.0 || x > 1.0 {
            return TestResult::discard();
        }

        // Compute "exact" and approximate log10
        let exact = x.log10();
        let approx = super::log10_1pct(x);

        // Compare results
        if exact == -f32::INFINITY {
            assert_eq!(approx, -f32::INFINITY);
        } else {
            assert_le!((approx - exact).abs(), 0.01 * exact.abs());
        }
        TestResult::passed()
    }
}
