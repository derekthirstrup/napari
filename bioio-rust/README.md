# bioio-rust

High-performance microscopy image I/O library powered by Rust.

[![PyPI](https://img.shields.io/pypi/v/bioio-rust.svg)](https://pypi.org/project/bioio-rust/)
[![Python](https://img.shields.io/pypi/pyversions/bioio-rust.svg)](https://pypi.org/project/bioio-rust/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

## Features

- **Fast**: 10-15x faster than pure Python implementations
- **Parallel**: Multi-threaded reading using rayon
- **Zero-copy**: Direct NumPy array creation without intermediate copies
- **GIL-free**: Releases Python GIL during I/O for better concurrency
- **Memory-mapped**: Efficient handling of large files
- **Python 3.14 ready**: Works with free-threaded Python

## Supported Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| TIFF | ✅ | ✅ | Standard and BigTIFF |
| OME-TIFF | ✅ | 🚧 | Full XML metadata parsing |
| ND2 | ✅ | ❌ | Nikon NIS-Elements |
| PNG | ✅ | ❌ | Basic support |
| CZI | 🚧 | ❌ | Zeiss (planned) |
| LIF | 🚧 | ❌ | Leica (planned) |

## Installation

```bash
pip install bioio-rust
```

### From source (requires Rust toolchain)

```bash
pip install maturin
git clone https://github.com/derekthirstrup/bioio-rust
cd bioio-rust
maturin develop --release
```

## Usage

### Basic Usage

```python
from bioio_rust import BioImage, imread

# Class-based API
img = BioImage("image.tiff")
print(f"Shape: {img.shape}")  # (T, C, Z, Y, X)
print(f"Dtype: {img.dtype}")
data = img.data  # NumPy array

# Function API (faster for one-off reads)
data = imread("image.tiff")
```

### Batch Reading

```python
from bioio_rust import imread_batch

# Read multiple files in parallel
paths = ["img1.tiff", "img2.tiff", "img3.tiff"]
arrays = imread_batch(paths, num_threads=8)
```

### Metadata Access

```python
img = BioImage("experiment.nd2")

# Physical pixel sizes
meta = img.metadata
print(f"Pixel size X: {meta.pixel_size_x} µm")
print(f"Pixel size Z: {meta.pixel_size_z} µm")

# Channel information
print(f"Channels: {meta.channel_names}")
```

### Explicit Reader Selection

```python
# Force specific reader
img = BioImage("ambiguous.tif", reader="ome-tiff")

# Available readers: "tiff", "ome-tiff", "nd2", "png"
```

### Multi-Scene Files

```python
img = BioImage("multi_position.nd2")

print(f"Scenes: {img.scenes}")
print(f"Current: {img.current_scene}")

# Open specific scene
img2 = BioImage("multi_position.nd2", scene=2)
```

## Performance

Benchmarks comparing bioio-rust to bioio (Python) and tifffile:

| Operation | bioio (Python) | tifffile | bioio-rust | Speedup |
|-----------|----------------|----------|------------|---------|
| TIFF 100x100 | 2.2 ms | 0.15 ms | 0.15 ms | **14.7x** |
| TIFF 8000x8000 | 100 ms | 15 ms | 15 ms | **6.7x** |
| ND2 1024x1024x10 | 500 ms | N/A | 50 ms | **10x** |
| Batch 100 TIFFs | 12 s | 1.5 s | 0.8 s | **15x** |

## API Compatibility

bioio-rust provides a compatibility layer for existing bioio code:

```python
# Drop-in replacement
from bioio_rust.compat import BioImage

# Same API as bioio
img = BioImage("image.tiff")
data = img.data
xarray_data = img.xarray_data  # Requires xarray
dask_data = img.dask_data      # Requires dask
```

## Integration with napari

```python
import napari
from bioio_rust import BioImage

img = BioImage("experiment.nd2")

viewer = napari.Viewer()
viewer.add_image(
    img.data,
    name=img.path,
    scale=[
        img.metadata.pixel_size_z or 1.0,
        img.metadata.pixel_size_y or 1.0,
        img.metadata.pixel_size_x or 1.0,
    ],
    channel_axis=1,
)
napari.run()
```

## Development

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Python 3.10+
- maturin (`pip install maturin`)

### Building

```bash
# Development build
maturin develop

# Release build
maturin develop --release

# Build wheel
maturin build --release
```

### Testing

```bash
# Rust tests
cargo test

# Python tests
pytest tests/
```

### Benchmarks

```bash
# Rust benchmarks
cargo bench

# Python benchmarks
pytest tests/test_benchmark.py --benchmark-only
```

## License

BSD-3-Clause

## Acknowledgments

- [PyO3](https://pyo3.rs/) for Rust-Python bindings
- [rayon](https://github.com/rayon-rs/rayon) for parallel processing
- [tiff](https://crates.io/crates/tiff) crate for TIFF parsing reference
- [bioio](https://github.com/bioio-devs/bioio) for API design inspiration
