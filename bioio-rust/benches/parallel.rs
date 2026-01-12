//! Benchmarks for parallel processing

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_chunk_processor(c: &mut Criterion) {
    use bioio_rust::parallel::ChunkProcessor;

    let mut group = c.benchmark_group("chunk_processor");

    for size in [100, 1000, 10000].iter() {
        let items: Vec<i32> = (0..*size).collect();

        group.bench_with_input(
            BenchmarkId::new("process", size),
            size,
            |b, _| {
                let processor = ChunkProcessor::new().unwrap();
                b.iter(|| {
                    let _results: Vec<i32> = black_box(
                        processor.process(items.clone(), |x| x * 2)
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_thread_scaling(c: &mut Criterion) {
    use bioio_rust::parallel::ChunkProcessor;

    let mut group = c.benchmark_group("thread_scaling");
    let items: Vec<i32> = (0..10000).collect();

    for threads in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            threads,
            |b, &threads| {
                let processor = ChunkProcessor::with_threads(threads).unwrap();
                b.iter(|| {
                    let _results: Vec<i32> = black_box(
                        processor.process(items.clone(), |x| {
                            // Simulate some work
                            let mut sum = 0i32;
                            for i in 0..100 {
                                sum = sum.wrapping_add(x.wrapping_mul(i));
                            }
                            sum
                        })
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_chunk_processor,
    bench_thread_scaling,
);

criterion_main!(benches);
