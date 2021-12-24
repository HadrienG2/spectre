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

/// log10 approximation for inputs in [0, 1] range with 5% absolute precision
pub fn log10_5pct(x: f32) -> f32 {
    // Handle strange and small inputs
    debug_assert!(!x.is_nan() && x >= 0.0 && x <= 1.0);
    if x.is_subnormal() || x == 0.0 {
        return -f32::INFINITY;
    }

    // Decompose input into its IEEE-754 components
    assert_eq!(f32::RADIX, 2);
    const NUM_F32_BITS: u32 = std::mem::size_of::<f32>() as u32 * 8;
    const NUM_MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
    const NUM_EXPONENT_BITS: u32 = NUM_F32_BITS - NUM_MANTISSA_BITS - 1;
    let mut x_bits = x.to_bits();
    let mantissa_bits = x_bits & ((1 << NUM_MANTISSA_BITS) - 1);
    x_bits >>= NUM_MANTISSA_BITS;
    let exponent_bits = x_bits & ((1 << NUM_EXPONENT_BITS) - 1);
    x_bits >>= NUM_EXPONENT_BITS;
    let sign_bit = x_bits & 1;

    // Use that to extract the exponent N, aka the integer part of x.log2(),
    // and produce w which is x normalized by 2^N
    const EXPONENT_ZERO_BITS: u32 = 2u32.pow(NUM_EXPONENT_BITS - 1) - 1;
    let exponent = exponent_bits as i32 - EXPONENT_ZERO_BITS as i32;
    let mut w_bits = sign_bit;
    w_bits = (w_bits << NUM_EXPONENT_BITS) | EXPONENT_ZERO_BITS;
    w_bits = (w_bits << NUM_MANTISSA_BITS) | mantissa_bits;
    let w = f32::from_bits(w_bits);

    // We use the Taylor expansion of log2 around 1 to approximate w.log2()
    let w_m1 = w - 1.0;
    const DEGREE: usize = 1 << 2;
    // FIXME: Can't have compiler const-fold this yet
    let coeffs: [_; DEGREE] = [
        1.0 / std::f32::consts::LN_2,
        -1.0 / (2.0 * std::f32::consts::LN_2),
        1.0 / (3.0 * std::f32::consts::LN_2),
        -1.0 / (4.0 * std::f32::consts::LN_2),
    ];
    let mut polynome = [0.0; DEGREE];
    let mut stride = 1;
    polynome[0] = w_m1;
    while stride < DEGREE {
        for i in 0..stride {
            // Initial:   w_m1
            // Stride 1:  w_m1  w_m1²
            // Stride 2:  w_m1  w_m1²  w_m1²w_m1  w_m1²²
            polynome[i + stride] = polynome[i] * polynome[stride - 1];
        }
        stride *= 2;
    }
    for (monome, coeff) in polynome.iter_mut().zip(coeffs.iter()) {
        *monome *= coeff;
    }
    let log2_w = polynome[..].iter().sum::<f32>();

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
    fn log10_5pct(mut x: f32) -> TestResult {
        // Only allow compatible input in [0, 1] range
        if !x.is_finite() {
            return TestResult::discard();
        }
        x = x.abs().fract();

        // Compute "exact" and approximate log10
        let exact = x.log10();
        let approx = super::log10_5pct(dbg!(x));

        // Compare results
        if exact == -f32::INFINITY {
            assert_eq!(approx, -f32::INFINITY);
        } else {
            assert_le!((dbg!(approx) - dbg!(exact)).abs(), 0.05);
        }
        TestResult::passed()
    }
}
