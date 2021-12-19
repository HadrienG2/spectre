//! Vectorized or auto-vectorizable computations

use std::{
    mem,
    ops::{Add, AddAssign},
};

// Native SIMD vector of f32s
// TODO: Use a proper SIMD library once available in stable Rust
#[cfg(not(target_feature = "avx"))]
#[repr(align(16))]
#[derive(Copy, Clone, Default)]
struct SimdF32([f32; 16 / mem::size_of::<f32>()]);
//
#[cfg(target_feature = "avx")]
#[repr(align(32))]
#[derive(Copy, Clone, Default)]
struct SimdF32([f32; 32 / mem::size_of::<f32>()]);
//
impl SimdF32 {
    /// Sum vector elements (TODO: leverage HADD)
    pub fn sum(self) -> f32 {
        self.0.into_iter().sum::<f32>()
    }
}
//
impl Add for SimdF32 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self {
        for (dest, src) in self.0.iter_mut().zip(rhs.0) {
            *dest += src;
        }
        self
    }
}
//
impl AddAssign for SimdF32 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Sum an array of f32s in a vectorizable manner
pub fn sum_f32(input: &[f32]) -> f32 {
    // Accumulation concurrency (TODO: tune through benchmarking)
    const CONCURRENCY: usize = 1 << 3;

    // Reinterprete input as a slice of aligned SIMD vectors + some extra floats
    let (peel, vectors, tail) = unsafe { input.align_to::<SimdF32>() };

    // Chunk the aligned SIMD data accordingly
    let chunks = vectors.chunks_exact(CONCURRENCY);
    let remainder = chunks.remainder();

    // Perform concurrent SIMD accumulation
    let mut accumulators = [SimdF32::default(); CONCURRENCY];
    for chunk in chunks {
        for (accumulator, &vector) in accumulators.iter_mut().zip(chunk) {
            *accumulator += vector;
        }
    }

    // Merge the SIMD accumulators into one
    let mut stride = CONCURRENCY / 2;
    while stride > 0 {
        for i in 0..stride {
            accumulators[i] += accumulators[i + stride];
        }
        stride /= 2;
    }
    let mut accumulator = accumulators[0];

    // Perform non-concurrent SIMD accumulation with remaining data
    for &vector in remainder {
        accumulator += vector;
    }

    // Reduce the SIMD vector into a scalar
    let simd_sum = accumulator.sum();

    // Finalize the sum
    let sum = |slice: &[f32]| slice.iter().sum::<f32>();
    sum(peel) + simd_sum + sum(tail)
}
