//! Benchmarks for bioio-rust readers

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;

fn create_test_tiff(width: u32, height: u32) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".tiff").unwrap();

    // Minimal TIFF header
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(b"II");  // Little endian
    data.extend_from_slice(&42u16.to_le_bytes());  // TIFF magic
    data.extend_from_slice(&8u32.to_le_bytes());  // First IFD offset

    // IFD with minimal tags
    let num_entries: u16 = 10;
    data.extend_from_slice(&num_entries.to_le_bytes());

    // Calculate image data offset (after IFD)
    let ifd_size = 2 + (num_entries as usize * 12) + 4;
    let image_offset = 8 + ifd_size;
    let image_size = (width * height) as usize;

    // Write tags
    write_tag(&mut data, 256, 4, 1, width as u64);  // ImageWidth
    write_tag(&mut data, 257, 4, 1, height as u64);  // ImageHeight
    write_tag(&mut data, 258, 3, 1, 8);  // BitsPerSample
    write_tag(&mut data, 259, 3, 1, 1);  // Compression (none)
    write_tag(&mut data, 262, 3, 1, 1);  // Photometric
    write_tag(&mut data, 273, 4, 1, image_offset as u64);  // StripOffsets
    write_tag(&mut data, 277, 3, 1, 1);  // SamplesPerPixel
    write_tag(&mut data, 278, 4, 1, height as u64);  // RowsPerStrip
    write_tag(&mut data, 279, 4, 1, image_size as u64);  // StripByteCounts
    write_tag(&mut data, 284, 3, 1, 1);  // PlanarConfig

    // Next IFD offset (0 = no more)
    data.extend_from_slice(&0u32.to_le_bytes());

    // Image data (grayscale gradient)
    for y in 0..height {
        for x in 0..width {
            let value = ((x + y) % 256) as u8;
            data.push(value);
        }
    }

    file.write_all(&data).unwrap();
    file.flush().unwrap();

    file
}

fn write_tag(data: &mut Vec<u8>, tag: u16, field_type: u16, count: u32, value: u64) {
    data.extend_from_slice(&tag.to_le_bytes());
    data.extend_from_slice(&field_type.to_le_bytes());
    data.extend_from_slice(&count.to_le_bytes());
    data.extend_from_slice(&(value as u32).to_le_bytes());
}

fn bench_tiff_read(c: &mut Criterion) {
    use bioio_rust::readers::{Reader, ReaderOptions, tiff::TiffReader};

    let mut group = c.benchmark_group("tiff_read");

    for size in [100, 500, 1000, 2000].iter() {
        let file = create_test_tiff(*size, *size);
        let path = file.path();

        group.bench_with_input(
            BenchmarkId::new("TiffReader", format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    let reader = TiffReader::open(path, &ReaderOptions::default()).unwrap();
                    let _data = black_box(reader.read_all().unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_tiff_parallel(c: &mut Criterion) {
    use bioio_rust::readers::{Reader, ReaderOptions, tiff::TiffReader};

    let mut group = c.benchmark_group("tiff_parallel");

    let file = create_test_tiff(2000, 2000);
    let path = file.path();

    for threads in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            threads,
            |b, &threads| {
                b.iter(|| {
                    let reader = TiffReader::open(path, &ReaderOptions::default()).unwrap();
                    let _data = black_box(reader.read_parallel(threads).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_format_detection(c: &mut Criterion) {
    use bioio_rust::format::ImageFormat;

    let file = create_test_tiff(100, 100);

    c.bench_function("format_detection", |b| {
        b.iter(|| {
            let _format = black_box(ImageFormat::detect(file.path()).unwrap());
        });
    });
}

criterion_group!(
    benches,
    bench_tiff_read,
    bench_tiff_parallel,
    bench_format_detection,
);

criterion_main!(benches);
