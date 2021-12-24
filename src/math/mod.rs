//! General-purpose math utilities

mod simd;

use realfft::num_complex::Complex;

pub use simd::sum_f32;

/// Interpolate a table of complex numbers into a series of complex numbers that
/// is approximately N times larger.
pub fn interpolate_c32(
    input: &[Complex<f32>],
    stride: usize,
) -> impl Iterator<Item = Complex<f32>> + '_ {
    assert!(input.len() > 0);
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
