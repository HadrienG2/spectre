use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use spectre::simd;

// NOTE: Due to current criterion limitations, you must rename main.rs into
//       lib.rs in order to be able to run this benchmark.
pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");
    for input_len in [8, 16, 32, 64, 128, 256, 1024, 2048, 4096] {
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
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
