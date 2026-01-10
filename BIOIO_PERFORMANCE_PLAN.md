# BioIO Performance Improvement Plan

## Executive Summary

BioIO exhibits significant performance regressions compared to native readers:
- **PNG**: ~231% slower than imageio (up to 2,778% for small images)
- **TIFF**: ~793% slower than tifffile (up to 1,386% for small images)

This document outlines a phased approach to address these issues, culminating in a potential Rust-based implementation for Python 3.14's free-threaded mode.

---

## Part 1: Root Cause Analysis

### Identified Performance Bottlenecks

| Issue | Impact | Severity |
|-------|--------|----------|
| Double read operations | 2x I/O overhead | Critical |
| Repeated tokenization (TIFF) | CPU-bound metadata parsing | Critical |
| Plugin discovery overhead | Costly for batch processing | High |
| Dask-based dimension detection | Extra read cycle before actual load | High |
| No reader caching | Re-initialization per file | Medium |

### Architectural Issues

1. **Generic Abstraction Tax**: BioIO's format-agnostic API requires probing files to determine dimensions, resulting in:
   - Initial dask-based lazy load to detect shape
   - Second non-dask load for actual data retrieval

2. **Plugin Discovery**: Reader selection mechanism iterates through all available plugins without caching resolution results.

3. **Metadata Over-Parsing**: TIFF tokenization happens repeatedly rather than being cached after first parse.

---

## Part 2: Short-Term Fixes (Python-Only)

### Phase 1: Quick Wins (1-2 weeks effort)

#### 1.1 Reader Caching Layer
```python
# Proposed: bioio/cache.py
from functools import lru_cache
from typing import Type
import threading

class ReaderCache:
    """Thread-safe reader resolution cache."""

    _lock = threading.Lock()
    _extension_cache: dict[str, Type['BioImage']] = {}
    _path_cache: dict[str, Type['BioImage']] = {}

    @classmethod
    def get_reader_for_extension(cls, ext: str) -> Type['BioImage'] | None:
        with cls._lock:
            return cls._extension_cache.get(ext)

    @classmethod
    def cache_reader(cls, ext: str, reader: Type['BioImage']) -> None:
        with cls._lock:
            cls._extension_cache[ext] = reader

    @classmethod
    def get_reader_for_path(cls, path: str) -> Type['BioImage'] | None:
        """Cache based on file signature/magic bytes."""
        # Use first 8 bytes + extension as key
        ...
```

#### 1.2 Explicit Reader Fast Path
```python
# When reader is explicitly specified, bypass all discovery
def open_image(path: str, reader: str | None = None):
    if reader is not None:
        # Direct instantiation - no discovery overhead
        return READER_REGISTRY[reader](path)

    # Check cache before discovery
    cached = ReaderCache.get_reader_for_extension(get_extension(path))
    if cached:
        return cached(path)

    # Fall back to discovery (slow path)
    return _discover_and_open(path)
```

#### 1.3 Lazy Metadata Loading
```python
class BioImage:
    def __init__(self, path: str, *, eager_metadata: bool = False):
        self._path = path
        self._metadata = None  # Lazy
        self._shape = None     # Lazy

        if eager_metadata:
            self._load_metadata()

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self._quick_shape_probe()
        return self._shape

    def _quick_shape_probe(self):
        """Format-specific fast shape detection without full parse."""
        # TIFF: Read first IFD only
        # PNG: Read IHDR chunk only
        # etc.
```

### Phase 2: Architecture Improvements (2-4 weeks effort)

#### 2.1 Eliminate Double Reads

**Current Flow:**
```
open() → dask_data (reads file) → detect dims → data (reads file again)
```

**Proposed Flow:**
```
open() → quick_probe() (reads header only) → lazy_data (deferred read)
                                           → eager_data (single full read)
```

#### 2.2 Format-Specific Optimizations

**TIFF:**
```python
class TiffReader:
    def __init__(self, path):
        self._tiff = tifffile.TiffFile(path)  # Single open
        self._pages_cache = {}

    def get_page(self, index: int) -> np.ndarray:
        if index not in self._pages_cache:
            self._pages_cache[index] = self._tiff.pages[index].asarray()
        return self._pages_cache[index]

    @property
    def shape(self):
        # No re-parse needed - TiffFile already has this
        return self._tiff.series[0].shape
```

**PNG:**
```python
class PngReader:
    def read_header_only(self) -> dict:
        """Read IHDR chunk for dimensions without decoding pixels."""
        with open(self._path, 'rb') as f:
            f.seek(8)  # Skip PNG signature
            # Read IHDR chunk
            length = struct.unpack('>I', f.read(4))[0]
            chunk_type = f.read(4)
            if chunk_type == b'IHDR':
                width = struct.unpack('>I', f.read(4))[0]
                height = struct.unpack('>I', f.read(4))[0]
                return {'width': width, 'height': height}
```

#### 2.3 Batch Processing Mode
```python
class BioImageBatch:
    """Optimized batch processing with shared reader instances."""

    def __init__(self, paths: list[str]):
        self._paths = paths
        self._reader_class = None  # Determined once

    def __iter__(self):
        # Determine reader from first file
        if self._reader_class is None:
            self._reader_class = determine_reader(self._paths[0])

        # Reuse reader class for all files
        for path in self._paths:
            yield self._reader_class(path)
```

---

## Part 3: Rust Migration Strategy for Python 3.14 Free-Threading

### Why Rust?

| Aspect | Python (Current) | Rust (Proposed) |
|--------|------------------|-----------------|
| GIL | Blocks true parallelism | N/A - native threads |
| Memory safety | Runtime checks | Compile-time guarantees |
| I/O performance | Good with asyncio | Excellent with tokio |
| CPU-bound work | Poor (GIL) | Excellent |
| Free-threaded Py3.14 | Experimental, overhead | Native integration via PyO3 |

### Python 3.14 Free-Threading Context

Python 3.14 (expected Oct 2025) includes experimental free-threaded mode (PEP 703):
- **Pro**: Removes GIL, enables true parallelism
- **Con**: ~40% single-thread overhead, C extension compatibility issues
- **Reality**: Most scientific libraries (NumPy, etc.) won't be compatible initially

**Rust via PyO3 sidesteps these issues entirely:**
- Native threads without GIL concerns
- Zero-copy array sharing with NumPy via `numpy` crate
- Works identically in GIL and free-threaded Python

### Proposed Architecture: `bioio-core` (Rust)

```
bioio-core (Rust)
├── src/
│   ├── lib.rs              # PyO3 module definition
│   ├── readers/
│   │   ├── mod.rs
│   │   ├── tiff.rs         # Via tiff crate
│   │   ├── png.rs          # Via png crate
│   │   ├── jpeg.rs         # Via jpeg-decoder
│   │   ├── zarr.rs         # Custom implementation
│   │   ├── ome_tiff.rs     # OME-TIFF with XML parsing
│   │   ├── czi.rs          # Zeiss CZI format
│   │   └── nd2.rs          # Nikon ND2 format
│   ├── writers/
│   │   ├── mod.rs
│   │   ├── tiff.rs
│   │   ├── ome_tiff.rs
│   │   └── zarr.rs
│   ├── metadata/
│   │   ├── mod.rs
│   │   ├── ome_xml.rs      # OME-XML parsing
│   │   └── dimensions.rs   # TCZYX handling
│   ├── parallel/
│   │   ├── mod.rs
│   │   ├── chunk_reader.rs # Parallel chunk loading
│   │   └── pool.rs         # Thread pool management
│   └── cache/
│       ├── mod.rs
│       ├── lru.rs          # LRU cache for tiles
│       └── mmap.rs         # Memory-mapped file support
├── python/
│   └── bioio_core/
│       ├── __init__.py
│       └── _core.pyi       # Type stubs
└── Cargo.toml
```

### Core Rust Implementation

#### Reader Trait
```rust
// src/readers/mod.rs
use numpy::PyArray;
use pyo3::prelude::*;

pub trait ImageReader: Send + Sync {
    fn open(path: &str) -> Result<Self, ReaderError> where Self: Sized;
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;
    fn read_region(&self, region: &Region) -> Result<ArrayD<u8>, ReaderError>;
    fn metadata(&self) -> &Metadata;
}

#[pyclass]
pub struct BioImage {
    reader: Box<dyn ImageReader>,
    path: String,
}

#[pymethods]
impl BioImage {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let reader = detect_and_open(path)?;
        Ok(Self { reader, path: path.to_string() })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.reader.shape().to_vec()
    }

    fn data<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray<u8, IxDyn>> {
        let arr = self.reader.read_region(&Region::Full)?;
        Ok(PyArray::from_array(py, &arr))
    }
}
```

#### Parallel TIFF Reader
```rust
// src/readers/tiff.rs
use rayon::prelude::*;
use tiff::decoder::{Decoder, DecodingResult};

pub struct TiffReader {
    path: String,
    ifd_offsets: Vec<u64>,
    shape: Vec<usize>,
    dtype: DType,
}

impl TiffReader {
    pub fn read_pages_parallel(&self, page_indices: &[usize]) -> Result<Vec<Array2<u8>>> {
        page_indices
            .par_iter()  // Parallel iteration via rayon
            .map(|&idx| self.read_single_page(idx))
            .collect()
    }

    pub fn read_tiles_parallel(&self, tile_indices: &[(usize, usize)]) -> Result<Vec<Array2<u8>>> {
        tile_indices
            .par_iter()
            .map(|&(row, col)| self.read_tile(row, col))
            .collect()
    }
}
```

#### Zero-Copy NumPy Integration
```rust
// src/lib.rs
use numpy::{PyArray, PyArrayMethods, ToPyArray};
use ndarray::ArrayD;

#[pyfunction]
fn read_tiff_fast<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray<u8, IxDyn>>> {
    // Release GIL during I/O
    let data: ArrayD<u8> = py.allow_threads(|| {
        let reader = TiffReader::open(path)?;
        reader.read_all()
    })?;

    // Zero-copy conversion to NumPy array
    Ok(data.to_pyarray_bound(py))
}
```

### Performance Targets

| Operation | Current (Python) | Target (Rust) | Improvement |
|-----------|------------------|---------------|-------------|
| TIFF 8000x8000 load | 100ms | 15ms | 6.7x |
| TIFF 100x100 load | 2.2ms | 0.15ms | 14.7x |
| PNG 8000x8000 load | 320ms | 120ms | 2.7x |
| Batch 1000 TIFFs | 120s | 8s | 15x |
| Parallel tile read | N/A (GIL) | 8 threads | 6-8x |

### Migration Path

#### Phase 1: Parallel Core (Month 1-2)
- Implement `bioio-core` with TIFF reader
- PyO3 bindings with zero-copy NumPy
- Benchmark against tifffile

#### Phase 2: Format Expansion (Month 3-4)
- Add PNG, JPEG readers
- OME-TIFF with XML metadata
- Zarr chunked reading

#### Phase 3: Advanced Formats (Month 5-6)
- CZI (Zeiss) format
- ND2 (Nikon) format
- LIF (Leica) format

#### Phase 4: Writers (Month 7-8)
- TIFF/BigTIFF writer with compression
- OME-TIFF writer
- Zarr writer with parallel chunk writing

#### Phase 5: Integration (Month 9-10)
- Replace bioio-python readers with Rust backend
- Maintain Python API compatibility
- Publish to PyPI with manylinux wheels

### Build & Distribution

```toml
# Cargo.toml
[package]
name = "bioio-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "bioio_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
numpy = "0.21"
ndarray = "0.15"
rayon = "1.10"
tiff = "0.9"
png = "0.17"
jpeg-decoder = "0.3"
quick-xml = "0.31"  # For OME-XML
memmap2 = "0.9"
thiserror = "1.0"

[build]
# maturin for Python packaging
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.4"]
build-backend = "maturin"

[project]
name = "bioio-core"
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
```

---

## Part 4: napari Integration

### Current napari Reader Flow
```
Viewer.open() → npe2.read() → plugin discovery → reader execution → layer creation
```

### Proposed Integration
```python
# napari_builtins/io/_read.py (modified)

def imread(filename: str) -> np.ndarray:
    ext = os.path.splitext(filename)[1].lower()

    # Fast path: Use Rust reader if available
    try:
        from bioio_core import read_fast
        return read_fast(filename)
    except ImportError:
        pass

    # Fallback to existing implementation
    if ext == '.npy':
        return np.load(filename)

    import imageio.v3 as iio
    return iio.imread(filename)
```

### Parallel Multi-File Loading
```python
# With Rust backend
from bioio_core import read_batch_parallel

def magic_imread(filenames, *, use_dask=None, stack=True):
    if len(filenames) > 1:
        # Use Rust parallel reader
        try:
            arrays = read_batch_parallel(filenames, num_threads=8)
            return np.stack(arrays) if stack else arrays
        except ImportError:
            pass

    # Fallback to dask-based implementation
    ...
```

---

## Part 5: Recommendations

### Immediate Actions (This Sprint)
1. **Implement reader caching** in bioio Python package
2. **Add explicit reader fast path** to bypass discovery
3. **Profile and fix double-read issue** in dimension detection

### Short-Term (1-2 Months)
1. **Format-specific header probing** for shape detection
2. **Batch processing API** with shared reader instances
3. **Benchmark suite** for continuous performance tracking

### Medium-Term (3-6 Months)
1. **Begin `bioio-core` Rust implementation** starting with TIFF
2. **PyO3 bindings** with zero-copy NumPy integration
3. **Parallel tile/chunk reading** for large images

### Long-Term (6-12 Months)
1. **Full format coverage** in Rust (CZI, ND2, LIF, etc.)
2. **Writer implementations** with parallel compression
3. **Replace bioio Python backend** with Rust core
4. **Python 3.14 free-threaded testing** and optimization

---

## Appendix A: Benchmark Script

```python
#!/usr/bin/env python
"""Benchmark bioio vs native readers."""

import time
import tempfile
import numpy as np
from pathlib import Path

def benchmark_readers():
    sizes = [(100, 100), (1000, 1000), (4000, 4000), (8000, 8000)]
    results = {}

    for h, w in sizes:
        # Create test image
        data = np.random.randint(0, 255, (h, w), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as f:
            import tifffile
            tifffile.imwrite(f.name, data)

            # Benchmark tifffile
            start = time.perf_counter()
            for _ in range(10):
                _ = tifffile.imread(f.name)
            tifffile_time = (time.perf_counter() - start) / 10

            # Benchmark bioio
            from bioio import BioImage
            start = time.perf_counter()
            for _ in range(10):
                img = BioImage(f.name)
                _ = img.data
            bioio_time = (time.perf_counter() - start) / 10

            results[f'{h}x{w}'] = {
                'tifffile': tifffile_time,
                'bioio': bioio_time,
                'ratio': bioio_time / tifffile_time
            }

            Path(f.name).unlink()

    return results

if __name__ == '__main__':
    results = benchmark_readers()
    for size, data in results.items():
        print(f"{size}: tifffile={data['tifffile']*1000:.2f}ms, "
              f"bioio={data['bioio']*1000:.2f}ms, "
              f"ratio={data['ratio']:.1f}x slower")
```

---

## Appendix B: Related Projects & Prior Art

| Project | Language | Notes |
|---------|----------|-------|
| [image-rs](https://github.com/image-rs/image) | Rust | General image I/O |
| [tiff](https://crates.io/crates/tiff) | Rust | TIFF decoder/encoder |
| [zarrs](https://github.com/LDeakin/zarrs) | Rust | Zarr v3 implementation |
| [pyo3](https://pyo3.rs/) | Rust | Python bindings |
| [polars](https://pola.rs/) | Rust | Example of successful Rust→Python |
| [ruff](https://github.com/astral-sh/ruff) | Rust | Example of 10-100x speedup |

---

## Conclusion

The bioio performance issues stem from architectural decisions that prioritize generality over speed. While Python-based optimizations can yield 2-5x improvements, a Rust-based core is the path to:

1. **True parallelism** without GIL constraints
2. **10-15x performance gains** for common operations
3. **Future-proofing** for Python 3.14 free-threading
4. **Memory safety** guarantees for scientific computing

The recommended approach is to pursue Python optimizations immediately while beginning Rust development in parallel, with full migration over 6-12 months.
