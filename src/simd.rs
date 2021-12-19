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
    /// Sum vector elements
    pub fn sum(&self) -> f32 {
        // NOTE: I tried smarter algorithms, but it would bust sum_f32 codegen.
        //       This is best left to explicit SIMD code, once possible.
        self.0.iter().sum::<f32>()
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

/// Sum an array of f32s
pub fn sum_f32(input: &[f32]) -> f32 {
    if cfg!(target_feature = "avx") {
        if input.len() < 16 {
            input.iter().sum::<f32>()
        } else if input.len() < 256 {
            sum_f32_impl::<1>(input)
        } else {
            sum_f32_impl::<2>(input)
        }
    } else {
        if input.len() < 32 {
            input.iter().sum::<f32>()
        } else if input.len() < 256 {
            sum_f32_impl::<1>(input)
        } else if input.len() < 1024 {
            sum_f32_impl::<2>(input)
        } else {
            sum_f32_impl::<4>(input)
        }
    }
}

/// Implementation of sum_f32 with tunable (power-of-2) SIMD op concurrency
pub fn sum_f32_impl<const CONCURRENCY: usize>(input: &[f32]) -> f32 {
    // Reinterprete input as a slice of aligned SIMD vectors + some extra floats
    let (peel, vectors, tail) = unsafe { input.align_to::<SimdF32>() };

    // Accumulate peel data
    let sum = |slice: &[f32]| slice.iter().sum::<f32>();
    let peel_sum = sum(peel);

    // Chunk the aligned SIMD data according to desired concurrency
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

    // Perform non-concurrent SIMD accumulation with remaining SIMD data
    for &vector in remainder {
        accumulator += vector;
    }

    // Reduce the SIMD vector into a scalar
    let simd_sum = accumulator.sum();

    // Accumulate tail data
    let tail_sum = sum(tail);

    // Deduce the final result
    peel_sum + simd_sum + tail_sum
}
