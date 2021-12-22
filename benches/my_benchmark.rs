use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use spectre::simd;

// NOTE: Due to current criterion limitations, you must rename main.rs into
//       lib.rs in order to be able to run this benchmark.
pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");
    for input_len in [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16 * 1024,
    ] {
        let input = std::iter::repeat(0.0).take(input_len).collect::<Box<[_]>>();
        group.throughput(Throughput::Bytes(
            (input_len * std::mem::size_of::<f32>()) as u64,
        ));
        group.bench_with_input(format!("naive/{}", input_len), &input, |b, input| {
            b.iter(|| input.iter().sum::<f32>());
        });
        group.bench_with_input(format!("optimized/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32(&input[..]));
        });
        /*
        // These benchmarks are useful for tuning simd::sum_f32, but require
        // making simd::sum_f32_impl public.
        group.bench_with_input(format!("simd1x1/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<1, 1>(&input[..]));
        });
        group.bench_with_input(format!("simd1x2/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<1, 2>(&input[..]));
        });
        group.bench_with_input(format!("simd1x4/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<1, 4>(&input[..]));
        });
        group.bench_with_input(format!("simd1x8/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<1, 8>(&input[..]));
        });
        group.bench_with_input(format!("simd2x1/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<2, 1>(&input[..]));
        });
        group.bench_with_input(format!("simd2x2/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<2, 2>(&input[..]));
        });
        group.bench_with_input(format!("simd2x4/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<2, 4>(&input[..]));
        });
        group.bench_with_input(format!("simd2x8/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<2, 8>(&input[..]));
        });
        group.bench_with_input(format!("simd4x1/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<4, 1>(&input[..]));
        });
        group.bench_with_input(format!("simd4x2/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<4, 2>(&input[..]));
        });
        group.bench_with_input(format!("simd4x4/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<4, 4>(&input[..]));
        });
        group.bench_with_input(format!("simd4x8/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<4, 8>(&input[..]));
        });
        group.bench_with_input(format!("simd8x1/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<8, 1>(&input[..]));
        });
        group.bench_with_input(format!("simd8x2/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<8, 2>(&input[..]));
        });
        group.bench_with_input(format!("simd8x4/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<8, 4>(&input[..]));
        });
        group.bench_with_input(format!("simd8x8/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<8, 8>(&input[..]));
        });
        group.bench_with_input(format!("simd16x1/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<16, 1>(&input[..]));
        });
        group.bench_with_input(format!("simd16x2/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<16, 2>(&input[..]));
        });
        group.bench_with_input(format!("simd16x4/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<16, 4>(&input[..]));
        });
        group.bench_with_input(format!("simd16x8/{}", input_len), &input, |b, input| {
            b.iter(|| simd::sum_f32_impl::<16, 8>(&input[..]));
        });
        */
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
