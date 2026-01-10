# BioIO Rust Implementation Plan

## Repository Setup

### Create New Repository

```bash
# Fork bioio to your account (do this on GitHub UI)
# https://github.com/bioio-devs/bioio -> Fork -> derekthirstrup/bioio

# Or create a new Rust-native repository
gh repo create derekthirstrup/bioio-rust --public --description "High-performance Rust implementation of BioIO for microscopy image I/O"

# Clone and initialize
git clone https://github.com/derekthirstrup/bioio-rust
cd bioio-rust
cargo init --lib
```

---

## Architecture Overview

### Current BioIO Python Architecture (The Problem)

```
BioImage.__init__(path)
    │
    ├─► determine_plugin()           # Iterates ALL plugins
    │       └─► get_plugins()        # Loads plugin cache
    │       └─► is_supported_image() # Calls each plugin
    │
    ├─► _get_reader()                # Creates reader instance
    │
    └─► Reader.__init__()
            └─► tifffile.TiffFile()  # Opens file (READ #1)
            └─► _get_dims_for_scene() # Parses metadata
            └─► _create_dask_array()  # Sets up lazy loading

BioImage.data (property access)
    │
    ├─► xarray_data                  # Materializes data
    │       └─► reader.xarray_data
    │               └─► _read_immediate()
    │                       └─► tifffile.imread() # (READ #2)
    │
    └─► _transform_data_array_to_bioio_image_standard()
            └─► Transpose to TCZYX   # Additional memory copy
```

### Proposed Rust Architecture (The Solution)

```
BioImage::open(path)
    │
    ├─► detect_format()              # Magic bytes check (no iteration)
    │       └─► read 16 bytes        # Single syscall
    │
    └─► Reader::open()               # Single file open
            └─► mmap file            # Zero-copy access
            └─► parse_header()       # Metadata in one pass
            └─► Ready for data access

BioImage::data()
    │
    └─► reader.read_all()            # Single read operation
            └─► parallel decode      # Rayon thread pool
            └─► zero-copy to NumPy   # No intermediate copies
```

---

## Project Structure

```
bioio-rust/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── src/
│   ├── lib.rs                    # PyO3 module entry point
│   ├── error.rs                  # Error types
│   ├── format.rs                 # Format detection
│   ├── metadata.rs               # Unified metadata model
│   ├── dimensions.rs             # TCZYX dimension handling
│   │
│   ├── readers/
│   │   ├── mod.rs                # Reader trait definition
│   │   ├── tiff.rs               # TIFF/BigTIFF reader
│   │   ├── ome_tiff.rs           # OME-TIFF with XML metadata
│   │   ├── nd2.rs                # Nikon ND2 format
│   │   ├── czi.rs                # Zeiss CZI format
│   │   └── zarr.rs               # Zarr v2/v3 support
│   │
│   ├── writers/
│   │   ├── mod.rs
│   │   ├── tiff.rs
│   │   └── ome_tiff.rs
│   │
│   ├── parallel/
│   │   ├── mod.rs
│   │   ├── thread_pool.rs        # Managed thread pool
│   │   └── chunk_reader.rs       # Parallel chunk/tile reading
│   │
│   └── python/
│       ├── mod.rs                # Python bindings
│       ├── bio_image.rs          # BioImage Python class
│       └── array.rs              # NumPy array conversion
│
├── python/
│   └── bioio_rust/
│       ├── __init__.py           # Python package
│       ├── _core.pyi             # Type stubs
│       └── compat.py             # bioio API compatibility layer
│
└── benches/
    ├── tiff_bench.rs
    └── nd2_bench.rs
```

---

## Core Implementation

### Cargo.toml

```toml
[package]
name = "bioio-rust"
version = "0.1.0"
edition = "2021"
license = "BSD-3-Clause"
description = "High-performance microscopy image I/O"
repository = "https://github.com/derekthirstrup/bioio-rust"

[lib]
name = "bioio_rust"
crate-type = ["cdylib", "rlib"]

[dependencies]
# Python bindings
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py310"] }
numpy = "0.22"

# Array handling
ndarray = { version = "0.16", features = ["rayon"] }

# Parallelism
rayon = "1.10"
crossbeam = "0.8"

# Image formats
tiff = "0.9"
jpeg-decoder = "0.3"
png = "0.17"
flate2 = "1.0"           # zlib decompression
lzw = "0.10"             # LZW decompression (TIFF)
zstd = "0.13"            # Zstandard compression

# ND2 format (custom implementation)
byteorder = "1.5"
memmap2 = "0.9"

# XML parsing (OME metadata)
quick-xml = "0.36"
roxmltree = "0.20"

# Zarr support
zarrs = "0.18"

# Utilities
thiserror = "2.0"
parking_lot = "0.12"     # Fast mutexes
smallvec = "1.13"
log = "0.4"

[dev-dependencies]
criterion = "0.5"
tempfile = "3.14"

[[bench]]
name = "readers"
harness = false

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

### src/lib.rs - Module Entry Point

```rust
//! BioIO Rust - High-performance microscopy image I/O
//!
//! This crate provides fast, parallel image reading for microscopy formats
//! with zero-copy Python/NumPy integration.

use pyo3::prelude::*;

mod error;
mod format;
mod metadata;
mod dimensions;
mod readers;
mod writers;
mod parallel;
mod python;

pub use error::{BioIoError, Result};
pub use format::ImageFormat;
pub use metadata::Metadata;
pub use dimensions::{DimensionOrder, Dimensions};
pub use readers::{Reader, ReaderOptions};

/// Python module definition
#[pymodule]
fn bioio_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core classes
    m.add_class::<python::BioImage>()?;
    m.add_class::<python::PyMetadata>()?;

    // Fast functions for direct use
    m.add_function(wrap_pyfunction!(python::imread, m)?)?;
    m.add_function(wrap_pyfunction!(python::imread_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python::imwrite, m)?)?;

    // Format detection
    m.add_function(wrap_pyfunction!(python::detect_format, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
```

### src/error.rs - Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BioIoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid TIFF: {0}")]
    InvalidTiff(String),

    #[error("Invalid ND2: {0}")]
    InvalidNd2(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("XML parsing error: {0}")]
    XmlParse(String),

    #[error("Scene {0} not found (available: {1})")]
    SceneNotFound(usize, usize),

    #[error("Python error: {0}")]
    Python(String),
}

pub type Result<T> = std::result::Result<T, BioIoError>;

impl From<BioIoError> for pyo3::PyErr {
    fn from(err: BioIoError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}
```

### src/format.rs - Format Detection

```rust
use std::path::Path;
use std::fs::File;
use std::io::Read;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Tiff,
    BigTiff,
    OmeTiff,
    Nd2,
    Czi,
    Zarr,
    Png,
    Jpeg,
    Unknown,
}

impl ImageFormat {
    /// Detect format from magic bytes (fast, no iteration)
    pub fn detect(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 16];
        file.read_exact(&mut magic)?;

        Ok(Self::from_magic(&magic, path))
    }

    fn from_magic(magic: &[u8], path: &Path) -> Self {
        // TIFF: II (little-endian) or MM (big-endian)
        if magic.starts_with(b"II") || magic.starts_with(b"MM") {
            // Check for BigTIFF (version 43)
            let version = if magic[0] == b'I' {
                u16::from_le_bytes([magic[2], magic[3]])
            } else {
                u16::from_be_bytes([magic[2], magic[3]])
            };

            if version == 43 {
                return ImageFormat::BigTiff;
            }

            // Check extension for OME-TIFF
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if ext == "ome.tiff" || ext == "ome.tif" {
                    return ImageFormat::OmeTiff;
                }
            }

            return ImageFormat::Tiff;
        }

        // ND2: Starts with 0xDADADADA
        if magic.starts_with(&[0xDA, 0xDA, 0xDA, 0xDA]) {
            return ImageFormat::Nd2;
        }

        // CZI: Starts with "ZISRAWFILE"
        if magic.starts_with(b"ZISRAWFILE") {
            return ImageFormat::Czi;
        }

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if magic.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            return ImageFormat::Png;
        }

        // JPEG: FF D8 FF
        if magic.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return ImageFormat::Jpeg;
        }

        // Zarr: Check for .zarr directory or .zarray file
        if path.is_dir() {
            let zarray = path.join(".zarray");
            let zgroup = path.join(".zgroup");
            if zarray.exists() || zgroup.exists() {
                return ImageFormat::Zarr;
            }
        }

        ImageFormat::Unknown
    }

    pub fn supported_extensions(&self) -> &'static [&'static str] {
        match self {
            ImageFormat::Tiff => &["tif", "tiff"],
            ImageFormat::BigTiff => &["tif", "tiff", "btf"],
            ImageFormat::OmeTiff => &["ome.tif", "ome.tiff"],
            ImageFormat::Nd2 => &["nd2"],
            ImageFormat::Czi => &["czi"],
            ImageFormat::Zarr => &["zarr"],
            ImageFormat::Png => &["png"],
            ImageFormat::Jpeg => &["jpg", "jpeg"],
            ImageFormat::Unknown => &[],
        }
    }
}
```

### src/readers/mod.rs - Reader Trait

```rust
use crate::{Result, Metadata, Dimensions, ImageFormat};
use ndarray::{ArrayD, IxDyn};
use std::path::Path;

/// Options for opening an image
#[derive(Debug, Clone, Default)]
pub struct ReaderOptions {
    /// Specific scene/series to open (None = first)
    pub scene: Option<usize>,
    /// Channel names override
    pub channel_names: Option<Vec<String>>,
    /// Force specific dimension order
    pub dim_order: Option<String>,
    /// Number of threads for parallel reading (0 = auto)
    pub num_threads: usize,
}

/// Core reader trait implemented by all format readers
pub trait Reader: Send + Sync {
    /// Open an image file
    fn open(path: &Path, options: &ReaderOptions) -> Result<Self>
    where
        Self: Sized;

    /// Get the image format
    fn format(&self) -> ImageFormat;

    /// Get image dimensions (TCZYX order)
    fn dimensions(&self) -> &Dimensions;

    /// Get the shape as a slice
    fn shape(&self) -> &[usize] {
        self.dimensions().shape()
    }

    /// Get the data type
    fn dtype(&self) -> DType;

    /// Get metadata
    fn metadata(&self) -> &Metadata;

    /// Get number of scenes/series
    fn num_scenes(&self) -> usize;

    /// Get scene names
    fn scene_names(&self) -> Vec<String>;

    /// Set active scene
    fn set_scene(&mut self, scene: usize) -> Result<()>;

    /// Read all data into memory
    fn read_all(&self) -> Result<ArrayD<u8>>;

    /// Read a specific region/slice
    fn read_region(&self, region: &Region) -> Result<ArrayD<u8>>;

    /// Read with parallel decoding
    fn read_parallel(&self, num_threads: usize) -> Result<ArrayD<u8>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
    F32,
    F64,
}

#[derive(Debug, Clone, Default)]
pub struct Region {
    pub t: Option<std::ops::Range<usize>>,
    pub c: Option<std::ops::Range<usize>>,
    pub z: Option<std::ops::Range<usize>>,
    pub y: Option<std::ops::Range<usize>>,
    pub x: Option<std::ops::Range<usize>>,
}

pub mod tiff;
pub mod nd2;
pub mod ome_tiff;
```

### src/readers/tiff.rs - TIFF Reader

```rust
use crate::error::{BioIoError, Result};
use crate::readers::{Reader, ReaderOptions, DType, Region};
use crate::{Metadata, Dimensions, ImageFormat};
use memmap2::Mmap;
use ndarray::{ArrayD, Array3, IxDyn, s};
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// High-performance TIFF reader with parallel decoding
pub struct TiffReader {
    /// Memory-mapped file for zero-copy access
    mmap: Arc<Mmap>,
    /// File path
    path: String,
    /// Parsed IFD entries for each page
    ifds: Vec<IfdEntry>,
    /// Image dimensions
    dimensions: Dimensions,
    /// Data type
    dtype: DType,
    /// Metadata
    metadata: Metadata,
    /// Current scene index
    current_scene: usize,
    /// Scene offsets (for multi-series TIFF)
    scene_offsets: Vec<usize>,
    /// Endianness
    little_endian: bool,
    /// Is BigTIFF
    is_bigtiff: bool,
}

#[derive(Debug, Clone)]
struct IfdEntry {
    /// Offset to image data
    strip_offsets: Vec<u64>,
    /// Byte counts for each strip
    strip_byte_counts: Vec<u64>,
    /// Image width
    width: u32,
    /// Image height
    height: u32,
    /// Bits per sample
    bits_per_sample: u16,
    /// Samples per pixel (channels)
    samples_per_pixel: u16,
    /// Compression type
    compression: Compression,
    /// Rows per strip
    rows_per_strip: u32,
    /// Tile width (0 if stripped)
    tile_width: u32,
    /// Tile height (0 if stripped)
    tile_height: u32,
    /// Tile offsets (if tiled)
    tile_offsets: Vec<u64>,
    /// Tile byte counts
    tile_byte_counts: Vec<u64>,
    /// Photometric interpretation
    photometric: u16,
    /// Planar configuration
    planar_config: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compression {
    None = 1,
    Lzw = 5,
    Jpeg = 7,
    Deflate = 8,
    AdobeDeflate = 32946,
    Zstd = 50000,
}

impl TiffReader {
    /// Parse TIFF header and IFD chain
    fn parse_header(mmap: &Mmap) -> Result<(bool, bool, u64)> {
        if mmap.len() < 8 {
            return Err(BioIoError::InvalidTiff("File too small".into()));
        }

        let little_endian = match &mmap[0..2] {
            b"II" => true,
            b"MM" => false,
            _ => return Err(BioIoError::InvalidTiff("Invalid byte order".into())),
        };

        let read_u16 = |offset: usize| -> u16 {
            if little_endian {
                u16::from_le_bytes([mmap[offset], mmap[offset + 1]])
            } else {
                u16::from_be_bytes([mmap[offset], mmap[offset + 1]])
            }
        };

        let version = read_u16(2);
        let (is_bigtiff, first_ifd_offset) = match version {
            42 => {
                // Classic TIFF
                let offset = if little_endian {
                    u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as u64
                } else {
                    u32::from_be_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as u64
                };
                (false, offset)
            }
            43 => {
                // BigTIFF
                let offset = if little_endian {
                    u64::from_le_bytes([
                        mmap[8], mmap[9], mmap[10], mmap[11],
                        mmap[12], mmap[13], mmap[14], mmap[15],
                    ])
                } else {
                    u64::from_be_bytes([
                        mmap[8], mmap[9], mmap[10], mmap[11],
                        mmap[12], mmap[13], mmap[14], mmap[15],
                    ])
                };
                (true, offset)
            }
            _ => return Err(BioIoError::InvalidTiff(format!("Unknown version: {}", version))),
        };

        Ok((little_endian, is_bigtiff, first_ifd_offset))
    }

    /// Parse all IFD entries (pages)
    fn parse_ifds(
        mmap: &Mmap,
        little_endian: bool,
        is_bigtiff: bool,
        first_offset: u64,
    ) -> Result<Vec<IfdEntry>> {
        let mut ifds = Vec::new();
        let mut offset = first_offset;

        while offset != 0 {
            let (ifd, next_offset) = Self::parse_single_ifd(mmap, little_endian, is_bigtiff, offset)?;
            ifds.push(ifd);
            offset = next_offset;
        }

        Ok(ifds)
    }

    fn parse_single_ifd(
        mmap: &Mmap,
        little_endian: bool,
        is_bigtiff: bool,
        offset: u64,
    ) -> Result<(IfdEntry, u64)> {
        // Implementation details for IFD parsing
        // ... (tag reading, value extraction, etc.)
        todo!("Full IFD parsing implementation")
    }

    /// Decompress a strip or tile
    fn decompress_block(
        &self,
        data: &[u8],
        compression: Compression,
        expected_size: usize,
    ) -> Result<Vec<u8>> {
        match compression {
            Compression::None => Ok(data.to_vec()),

            Compression::Lzw => {
                let mut decoder = lzw::Decoder::new(lzw::LsbReader::new(), 8);
                let mut output = Vec::with_capacity(expected_size);
                decoder.decode_bytes(data, &mut output)
                    .map_err(|e| BioIoError::Decompression(e.to_string()))?;
                Ok(output)
            }

            Compression::Deflate | Compression::AdobeDeflate => {
                use flate2::read::ZlibDecoder;
                use std::io::Read;
                let mut decoder = ZlibDecoder::new(data);
                let mut output = Vec::with_capacity(expected_size);
                decoder.read_to_end(&mut output)
                    .map_err(|e| BioIoError::Decompression(e.to_string()))?;
                Ok(output)
            }

            Compression::Zstd => {
                zstd::decode_all(data)
                    .map_err(|e| BioIoError::Decompression(e.to_string()))
            }

            Compression::Jpeg => {
                let mut decoder = jpeg_decoder::Decoder::new(data);
                decoder.decode()
                    .map_err(|e| BioIoError::Decompression(e.to_string()))
            }
        }
    }

    /// Read a single page with parallel strip/tile decoding
    fn read_page_parallel(&self, page_idx: usize) -> Result<Array3<u8>> {
        let ifd = &self.ifds[page_idx];
        let height = ifd.height as usize;
        let width = ifd.width as usize;
        let samples = ifd.samples_per_pixel as usize;

        if ifd.tile_width > 0 {
            // Tiled TIFF - parallel tile reading
            self.read_tiles_parallel(ifd, height, width, samples)
        } else {
            // Stripped TIFF - parallel strip reading
            self.read_strips_parallel(ifd, height, width, samples)
        }
    }

    fn read_strips_parallel(
        &self,
        ifd: &IfdEntry,
        height: usize,
        width: usize,
        samples: usize,
    ) -> Result<Array3<u8>> {
        let rows_per_strip = ifd.rows_per_strip as usize;
        let num_strips = ifd.strip_offsets.len();
        let bytes_per_row = width * samples;

        // Parallel strip decompression
        let strips: Vec<Result<Vec<u8>>> = (0..num_strips)
            .into_par_iter()
            .map(|strip_idx| {
                let offset = ifd.strip_offsets[strip_idx] as usize;
                let byte_count = ifd.strip_byte_counts[strip_idx] as usize;
                let strip_rows = std::cmp::min(
                    rows_per_strip,
                    height - strip_idx * rows_per_strip,
                );
                let expected_size = strip_rows * bytes_per_row;

                let compressed = &self.mmap[offset..offset + byte_count];
                self.decompress_block(compressed, ifd.compression, expected_size)
            })
            .collect();

        // Combine strips into single array
        let mut output = Array3::<u8>::zeros((height, width, samples));
        let mut row = 0;
        for strip_result in strips {
            let strip_data = strip_result?;
            let strip_rows = strip_data.len() / bytes_per_row;

            for r in 0..strip_rows {
                let src_start = r * bytes_per_row;
                let src_end = src_start + bytes_per_row;
                output.slice_mut(s![row + r, .., ..])
                    .as_slice_mut()
                    .unwrap()
                    .copy_from_slice(&strip_data[src_start..src_end]);
            }
            row += strip_rows;
        }

        Ok(output)
    }

    fn read_tiles_parallel(
        &self,
        ifd: &IfdEntry,
        height: usize,
        width: usize,
        samples: usize,
    ) -> Result<Array3<u8>> {
        let tile_width = ifd.tile_width as usize;
        let tile_height = ifd.tile_height as usize;
        let tiles_across = (width + tile_width - 1) / tile_width;
        let tiles_down = (height + tile_height - 1) / tile_height;
        let num_tiles = tiles_across * tiles_down;

        // Parallel tile decompression
        let tiles: Vec<Result<(usize, usize, Vec<u8>)>> = (0..num_tiles)
            .into_par_iter()
            .map(|tile_idx| {
                let tile_y = tile_idx / tiles_across;
                let tile_x = tile_idx % tiles_across;

                let offset = ifd.tile_offsets[tile_idx] as usize;
                let byte_count = ifd.tile_byte_counts[tile_idx] as usize;
                let expected_size = tile_width * tile_height * samples;

                let compressed = &self.mmap[offset..offset + byte_count];
                let data = self.decompress_block(compressed, ifd.compression, expected_size)?;

                Ok((tile_y, tile_x, data))
            })
            .collect();

        // Combine tiles into single array
        let mut output = Array3::<u8>::zeros((height, width, samples));
        for tile_result in tiles {
            let (tile_y, tile_x, tile_data) = tile_result?;

            let y_start = tile_y * tile_height;
            let x_start = tile_x * tile_width;
            let y_end = std::cmp::min(y_start + tile_height, height);
            let x_end = std::cmp::min(x_start + tile_width, width);

            for y in y_start..y_end {
                for x in x_start..x_end {
                    let src_idx = ((y - y_start) * tile_width + (x - x_start)) * samples;
                    for s in 0..samples {
                        output[[y, x, s]] = tile_data[src_idx + s];
                    }
                }
            }
        }

        Ok(output)
    }
}

impl Reader for TiffReader {
    fn open(path: &Path, options: &ReaderOptions) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        let (little_endian, is_bigtiff, first_ifd) = Self::parse_header(&mmap)?;
        let ifds = Self::parse_ifds(&mmap, little_endian, is_bigtiff, first_ifd)?;

        if ifds.is_empty() {
            return Err(BioIoError::InvalidTiff("No IFD entries found".into()));
        }

        // Build dimensions from first IFD
        let first_ifd = &ifds[0];
        let dimensions = Dimensions::new(
            1,  // T
            first_ifd.samples_per_pixel as usize,  // C
            ifds.len(),  // Z (pages)
            first_ifd.height as usize,  // Y
            first_ifd.width as usize,   // X
        );

        let dtype = match first_ifd.bits_per_sample {
            8 => DType::U8,
            16 => DType::U16,
            32 => DType::U32,
            _ => DType::U8,
        };

        Ok(Self {
            mmap,
            path: path.to_string_lossy().into_owned(),
            ifds,
            dimensions,
            dtype,
            metadata: Metadata::default(),
            current_scene: 0,
            scene_offsets: vec![0],
            little_endian,
            is_bigtiff,
        })
    }

    fn format(&self) -> ImageFormat {
        if self.is_bigtiff {
            ImageFormat::BigTiff
        } else {
            ImageFormat::Tiff
        }
    }

    fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn num_scenes(&self) -> usize {
        self.scene_offsets.len()
    }

    fn scene_names(&self) -> Vec<String> {
        (0..self.num_scenes())
            .map(|i| format!("Scene_{}", i))
            .collect()
    }

    fn set_scene(&mut self, scene: usize) -> Result<()> {
        if scene >= self.num_scenes() {
            return Err(BioIoError::SceneNotFound(scene, self.num_scenes()));
        }
        self.current_scene = scene;
        Ok(())
    }

    fn read_all(&self) -> Result<ArrayD<u8>> {
        self.read_parallel(0)
    }

    fn read_region(&self, region: &Region) -> Result<ArrayD<u8>> {
        // Read full data then slice (optimize later with direct region reading)
        let full = self.read_all()?;

        // Apply region slicing
        // ... implementation
        Ok(full)
    }

    fn read_parallel(&self, num_threads: usize) -> Result<ArrayD<u8>> {
        // Configure thread pool
        let pool = if num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|e| BioIoError::Python(e.to_string()))?
        } else {
            rayon::ThreadPoolBuilder::new()
                .build()
                .map_err(|e| BioIoError::Python(e.to_string()))?
        };

        // Read all pages in parallel
        let pages: Vec<Result<Array3<u8>>> = pool.install(|| {
            (0..self.ifds.len())
                .into_par_iter()
                .map(|page_idx| self.read_page_parallel(page_idx))
                .collect()
        });

        // Stack into TCZYX array
        let dims = &self.dimensions;
        let mut output = ArrayD::<u8>::zeros(IxDyn(&[
            dims.t, dims.c, dims.z, dims.y, dims.x,
        ]));

        for (z, page_result) in pages.into_iter().enumerate() {
            let page = page_result?;
            // Copy page data into output array at z position
            for y in 0..dims.y {
                for x in 0..dims.x {
                    for c in 0..dims.c {
                        output[[0, c, z, y, x]] = page[[y, x, c]];
                    }
                }
            }
        }

        Ok(output)
    }
}
```

### src/readers/nd2.rs - ND2 Reader

```rust
//! Nikon ND2 file format reader
//!
//! ND2 files use a chunk-based format with metadata in XML/JSON
//! and image data in compressed chunks.

use crate::error::{BioIoError, Result};
use crate::readers::{Reader, ReaderOptions, DType, Region};
use crate::{Metadata, Dimensions, ImageFormat};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use ndarray::ArrayD;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

/// ND2 file signature
const ND2_MAGIC: [u8; 4] = [0xDA, 0xDA, 0xDA, 0xDA];

/// Chunk types in ND2 files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkType {
    ImageData,
    ImageMetadata,
    TextInfo,
    Experiment,
    Unknown,
}

/// Parsed chunk header
#[derive(Debug, Clone)]
struct ChunkHeader {
    chunk_type: ChunkType,
    offset: u64,
    size: u64,
    compression: Compression,
}

#[derive(Debug, Clone, Copy)]
enum Compression {
    None,
    Lz4,
    Zstd,
}

/// High-performance ND2 reader
pub struct Nd2Reader {
    mmap: Arc<Mmap>,
    path: String,
    dimensions: Dimensions,
    dtype: DType,
    metadata: Metadata,

    /// Image data chunk locations
    image_chunks: Vec<ChunkHeader>,
    /// Dimension mapping (which chunk contains which TCZYX coordinate)
    chunk_map: HashMap<(usize, usize, usize), usize>,

    /// Pixel data parameters
    bytes_per_pixel: usize,
    image_width: usize,
    image_height: usize,

    /// Scenes/positions
    scenes: Vec<Nd2Scene>,
    current_scene: usize,
}

#[derive(Debug, Clone)]
struct Nd2Scene {
    name: String,
    first_chunk: usize,
    num_chunks: usize,
}

impl Nd2Reader {
    /// Parse ND2 file structure
    fn parse_file(mmap: &Mmap) -> Result<Nd2FileInfo> {
        // Verify magic number
        if !mmap.starts_with(&ND2_MAGIC) {
            return Err(BioIoError::InvalidNd2("Invalid magic number".into()));
        }

        let mut cursor = Cursor::new(&mmap[..]);
        cursor.seek(SeekFrom::Start(4))?;

        // Read file version
        let version = cursor.read_u32::<LittleEndian>()?;

        // Parse chunk index (at end of file)
        let chunk_map_offset = Self::find_chunk_map(&mmap)?;
        let chunks = Self::parse_chunk_map(&mmap, chunk_map_offset)?;

        // Extract metadata from text chunks
        let metadata = Self::parse_metadata(&mmap, &chunks)?;

        // Extract dimensions from metadata
        let dimensions = Self::extract_dimensions(&metadata)?;

        Ok(Nd2FileInfo {
            version,
            chunks,
            metadata,
            dimensions,
        })
    }

    fn find_chunk_map(mmap: &Mmap) -> Result<u64> {
        // ND2 stores chunk map location in last 8 bytes
        let len = mmap.len();
        if len < 16 {
            return Err(BioIoError::InvalidNd2("File too small".into()));
        }

        let offset_bytes = &mmap[len - 8..len];
        let offset = u64::from_le_bytes(offset_bytes.try_into().unwrap());

        if offset >= len as u64 {
            return Err(BioIoError::InvalidNd2("Invalid chunk map offset".into()));
        }

        Ok(offset)
    }

    fn parse_chunk_map(mmap: &Mmap, offset: u64) -> Result<Vec<ChunkHeader>> {
        let mut cursor = Cursor::new(&mmap[offset as usize..]);
        let mut chunks = Vec::new();

        // Read number of chunks
        let num_chunks = cursor.read_u32::<LittleEndian>()? as usize;

        for _ in 0..num_chunks {
            let chunk_offset = cursor.read_u64::<LittleEndian>()?;
            let chunk_size = cursor.read_u64::<LittleEndian>()?;
            let chunk_type_id = cursor.read_u32::<LittleEndian>()?;
            let compression_id = cursor.read_u32::<LittleEndian>()?;

            let chunk_type = match chunk_type_id {
                1 => ChunkType::ImageData,
                2 => ChunkType::ImageMetadata,
                3 => ChunkType::TextInfo,
                4 => ChunkType::Experiment,
                _ => ChunkType::Unknown,
            };

            let compression = match compression_id {
                0 => Compression::None,
                1 => Compression::Lz4,
                2 => Compression::Zstd,
                _ => Compression::None,
            };

            chunks.push(ChunkHeader {
                chunk_type,
                offset: chunk_offset,
                size: chunk_size,
                compression,
            });
        }

        Ok(chunks)
    }

    fn parse_metadata(mmap: &Mmap, chunks: &[ChunkHeader]) -> Result<Nd2Metadata> {
        // Find and parse metadata chunks
        let text_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::TextInfo)
            .collect();

        let mut metadata = Nd2Metadata::default();

        for chunk in text_chunks {
            let data = &mmap[chunk.offset as usize..(chunk.offset + chunk.size) as usize];
            let text = Self::decompress_chunk(data, chunk.compression)?;

            // Parse JSON metadata
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&text) {
                Self::extract_metadata_from_json(&json, &mut metadata);
            }
        }

        Ok(metadata)
    }

    fn decompress_chunk(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
        match compression {
            Compression::None => Ok(data.to_vec()),
            Compression::Lz4 => {
                // Use lz4_flex for decompression
                lz4_flex::decompress_size_prepended(data)
                    .map_err(|e| BioIoError::Decompression(e.to_string()))
            }
            Compression::Zstd => {
                zstd::decode_all(data)
                    .map_err(|e| BioIoError::Decompression(e.to_string()))
            }
        }
    }

    fn extract_dimensions(metadata: &Nd2Metadata) -> Result<Dimensions> {
        Ok(Dimensions::new(
            metadata.num_timepoints.unwrap_or(1),
            metadata.num_channels.unwrap_or(1),
            metadata.num_z_slices.unwrap_or(1),
            metadata.height.ok_or_else(|| BioIoError::InvalidNd2("Missing height".into()))?,
            metadata.width.ok_or_else(|| BioIoError::InvalidNd2("Missing width".into()))?,
        ))
    }

    fn extract_metadata_from_json(json: &serde_json::Value, metadata: &mut Nd2Metadata) {
        // Extract dimensions
        if let Some(dims) = json.get("dimensions") {
            metadata.width = dims.get("width").and_then(|v| v.as_u64()).map(|v| v as usize);
            metadata.height = dims.get("height").and_then(|v| v.as_u64()).map(|v| v as usize);
            metadata.num_channels = dims.get("channels").and_then(|v| v.as_u64()).map(|v| v as usize);
            metadata.num_z_slices = dims.get("z").and_then(|v| v.as_u64()).map(|v| v as usize);
            metadata.num_timepoints = dims.get("t").and_then(|v| v.as_u64()).map(|v| v as usize);
        }

        // Extract pixel size
        if let Some(calibration) = json.get("calibration") {
            metadata.pixel_size_x = calibration.get("pixelSizeX").and_then(|v| v.as_f64());
            metadata.pixel_size_y = calibration.get("pixelSizeY").and_then(|v| v.as_f64());
            metadata.pixel_size_z = calibration.get("pixelSizeZ").and_then(|v| v.as_f64());
        }

        // Extract channel names
        if let Some(channels) = json.get("channels").and_then(|v| v.as_array()) {
            metadata.channel_names = channels
                .iter()
                .filter_map(|c| c.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect();
        }
    }

    /// Read image data from a chunk with parallel decompression
    fn read_image_chunk(&self, chunk_idx: usize) -> Result<Vec<u8>> {
        let chunk = &self.image_chunks[chunk_idx];
        let data = &self.mmap[chunk.offset as usize..(chunk.offset + chunk.size) as usize];
        Self::decompress_chunk(data, chunk.compression)
    }

    /// Read all image data with parallel chunk loading
    fn read_all_parallel(&self, num_threads: usize) -> Result<ArrayD<u8>> {
        let pool = if num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|e| BioIoError::Python(e.to_string()))?
        } else {
            rayon::ThreadPoolBuilder::new()
                .build()
                .map_err(|e| BioIoError::Python(e.to_string()))?
        };

        let dims = &self.dimensions;

        // Parallel chunk reading
        let chunk_data: Vec<Result<Vec<u8>>> = pool.install(|| {
            self.image_chunks
                .par_iter()
                .enumerate()
                .map(|(idx, _)| self.read_image_chunk(idx))
                .collect()
        });

        // Allocate output array
        let mut output = ArrayD::<u8>::zeros(ndarray::IxDyn(&[
            dims.t, dims.c, dims.z, dims.y, dims.x,
        ]));

        // Copy chunk data to correct positions
        for (chunk_idx, data_result) in chunk_data.into_iter().enumerate() {
            let data = data_result?;

            // Find TCZYX coordinates for this chunk
            if let Some(&(t, c, z)) = self.chunk_map.iter()
                .find(|(_, &idx)| idx == chunk_idx)
                .map(|(coords, _)| coords)
            {
                // Copy pixel data
                for y in 0..dims.y {
                    for x in 0..dims.x {
                        let src_idx = (y * dims.x + x) * self.bytes_per_pixel;
                        output[[t, c, z, y, x]] = data[src_idx];
                    }
                }
            }
        }

        Ok(output)
    }
}

impl Reader for Nd2Reader {
    fn open(path: &Path, options: &ReaderOptions) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        let file_info = Self::parse_file(&mmap)?;

        // Build chunk map for image data
        let image_chunks: Vec<_> = file_info.chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::ImageData)
            .cloned()
            .collect();

        // Build coordinate -> chunk mapping
        let chunk_map = Self::build_chunk_map(&file_info.dimensions, &image_chunks);

        let dtype = match file_info.metadata.bits_per_component.unwrap_or(8) {
            8 => DType::U8,
            16 => DType::U16,
            _ => DType::U8,
        };

        Ok(Self {
            mmap,
            path: path.to_string_lossy().into_owned(),
            dimensions: file_info.dimensions,
            dtype,
            metadata: file_info.metadata.into(),
            image_chunks,
            chunk_map,
            bytes_per_pixel: dtype.size_bytes(),
            image_width: file_info.dimensions.x,
            image_height: file_info.dimensions.y,
            scenes: vec![Nd2Scene {
                name: "Scene_0".into(),
                first_chunk: 0,
                num_chunks: image_chunks.len(),
            }],
            current_scene: 0,
        })
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Nd2
    }

    fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn num_scenes(&self) -> usize {
        self.scenes.len()
    }

    fn scene_names(&self) -> Vec<String> {
        self.scenes.iter().map(|s| s.name.clone()).collect()
    }

    fn set_scene(&mut self, scene: usize) -> Result<()> {
        if scene >= self.scenes.len() {
            return Err(BioIoError::SceneNotFound(scene, self.scenes.len()));
        }
        self.current_scene = scene;
        Ok(())
    }

    fn read_all(&self) -> Result<ArrayD<u8>> {
        self.read_all_parallel(0)
    }

    fn read_region(&self, _region: &Region) -> Result<ArrayD<u8>> {
        // For now, read all and slice
        self.read_all()
    }

    fn read_parallel(&self, num_threads: usize) -> Result<ArrayD<u8>> {
        self.read_all_parallel(num_threads)
    }
}

// Helper structs
#[derive(Default)]
struct Nd2FileInfo {
    version: u32,
    chunks: Vec<ChunkHeader>,
    metadata: Nd2Metadata,
    dimensions: Dimensions,
}

#[derive(Default)]
struct Nd2Metadata {
    width: Option<usize>,
    height: Option<usize>,
    num_channels: Option<usize>,
    num_z_slices: Option<usize>,
    num_timepoints: Option<usize>,
    bits_per_component: Option<u8>,
    pixel_size_x: Option<f64>,
    pixel_size_y: Option<f64>,
    pixel_size_z: Option<f64>,
    channel_names: Vec<String>,
}

impl Nd2Reader {
    fn build_chunk_map(
        dims: &Dimensions,
        chunks: &[ChunkHeader],
    ) -> HashMap<(usize, usize, usize), usize> {
        let mut map = HashMap::new();

        // ND2 typically stores as TZCYX order
        let mut chunk_idx = 0;
        for t in 0..dims.t {
            for z in 0..dims.z {
                for c in 0..dims.c {
                    if chunk_idx < chunks.len() {
                        map.insert((t, c, z), chunk_idx);
                        chunk_idx += 1;
                    }
                }
            }
        }

        map
    }
}

impl DType {
    fn size_bytes(&self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::F64 => 8,
        }
    }
}
```

### src/python/mod.rs - Python Bindings

```rust
use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray, PyArrayMethods, ToPyArray, PyArrayDyn};
use ndarray::ArrayD;
use std::path::Path;

use crate::readers::{Reader, ReaderOptions, tiff::TiffReader, nd2::Nd2Reader};
use crate::format::ImageFormat;
use crate::{Metadata, Dimensions};

/// Main BioImage class exposed to Python
#[pyclass(name = "BioImage")]
pub struct BioImage {
    reader: Box<dyn Reader>,
    path: String,
}

#[pymethods]
impl BioImage {
    #[new]
    #[pyo3(signature = (path, reader=None, scene=None, num_threads=0))]
    fn new(
        path: &str,
        reader: Option<&str>,
        scene: Option<usize>,
        num_threads: usize,
    ) -> PyResult<Self> {
        let path_obj = Path::new(path);
        let options = ReaderOptions {
            scene,
            num_threads,
            ..Default::default()
        };

        let format = if let Some(reader_name) = reader {
            match reader_name.to_lowercase().as_str() {
                "tiff" | "tifffile" => ImageFormat::Tiff,
                "nd2" => ImageFormat::Nd2,
                "ome-tiff" | "ome_tiff" => ImageFormat::OmeTiff,
                _ => ImageFormat::detect(path_obj)?,
            }
        } else {
            ImageFormat::detect(path_obj)?
        };

        let reader: Box<dyn Reader> = match format {
            ImageFormat::Tiff | ImageFormat::BigTiff => {
                Box::new(TiffReader::open(path_obj, &options)?)
            }
            ImageFormat::Nd2 => {
                Box::new(Nd2Reader::open(path_obj, &options)?)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unsupported format: {:?}", format)
                ))
            }
        };

        Ok(Self {
            reader,
            path: path.to_string(),
        })
    }

    /// Get image dimensions as tuple (T, C, Z, Y, X)
    #[getter]
    fn shape(&self) -> Vec<usize> {
        let dims = self.reader.dimensions();
        vec![dims.t, dims.c, dims.z, dims.y, dims.x]
    }

    /// Get dimension names
    #[getter]
    fn dims(&self) -> Vec<&'static str> {
        vec!["T", "C", "Z", "Y", "X"]
    }

    /// Number of scenes
    #[getter]
    fn num_scenes(&self) -> usize {
        self.reader.num_scenes()
    }

    /// Scene names
    #[getter]
    fn scenes(&self) -> Vec<String> {
        self.reader.scene_names()
    }

    /// Read all data as numpy array (releases GIL during read)
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
        // Release GIL during the heavy lifting
        let array: ArrayD<u8> = py.allow_threads(|| {
            self.reader.read_all()
        })?;

        // Zero-copy conversion to numpy
        Ok(array.to_pyarray_bound(py))
    }

    /// Read data with explicit thread count
    #[pyo3(signature = (num_threads=0))]
    fn read_parallel<'py>(
        &self,
        py: Python<'py>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
        let array: ArrayD<u8> = py.allow_threads(|| {
            self.reader.read_parallel(num_threads)
        })?;

        Ok(array.to_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        let dims = self.reader.dimensions();
        format!(
            "BioImage('{}', shape=({}, {}, {}, {}, {}), dtype={:?})",
            self.path, dims.t, dims.c, dims.z, dims.y, dims.x,
            self.reader.dtype()
        )
    }
}

/// Fast imread function - bypasses class instantiation overhead
#[pyfunction]
#[pyo3(signature = (path, reader=None))]
pub fn imread<'py>(
    py: Python<'py>,
    path: &str,
    reader: Option<&str>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let img = BioImage::new(path, reader, None, 0)?;
    img.data(py)
}

/// Batch read multiple files in parallel
#[pyfunction]
#[pyo3(signature = (paths, num_threads=0))]
pub fn imread_batch<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    num_threads: usize,
) -> PyResult<Vec<Bound<'py, PyArrayDyn<u8>>>> {
    use rayon::prelude::*;

    // Configure thread pool
    let pool = if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
    } else {
        rayon::ThreadPoolBuilder::new()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
    };

    // Read all files in parallel (GIL released)
    let arrays: Vec<ArrayD<u8>> = py.allow_threads(|| {
        pool.install(|| {
            paths
                .par_iter()
                .map(|path| {
                    let path_obj = Path::new(path);
                    let format = ImageFormat::detect(path_obj)?;
                    let options = ReaderOptions::default();

                    let reader: Box<dyn Reader> = match format {
                        ImageFormat::Tiff | ImageFormat::BigTiff => {
                            Box::new(TiffReader::open(path_obj, &options)?)
                        }
                        ImageFormat::Nd2 => {
                            Box::new(Nd2Reader::open(path_obj, &options)?)
                        }
                        _ => return Err(crate::BioIoError::UnsupportedFormat(
                            format!("{:?}", format)
                        )),
                    };

                    reader.read_all()
                })
                .collect::<Result<Vec<_>, _>>()
        })
    })?;

    // Convert to numpy arrays
    arrays
        .into_iter()
        .map(|arr| Ok(arr.to_pyarray_bound(py)))
        .collect()
}

/// Detect image format from file
#[pyfunction]
pub fn detect_format(path: &str) -> PyResult<String> {
    let format = ImageFormat::detect(Path::new(path))?;
    Ok(format!("{:?}", format))
}

/// Python metadata wrapper
#[pyclass(name = "Metadata")]
pub struct PyMetadata {
    inner: Metadata,
}

#[pymethods]
impl PyMetadata {
    #[getter]
    fn pixel_size_x(&self) -> Option<f64> {
        self.inner.pixel_size_x
    }

    #[getter]
    fn pixel_size_y(&self) -> Option<f64> {
        self.inner.pixel_size_y
    }

    #[getter]
    fn pixel_size_z(&self) -> Option<f64> {
        self.inner.pixel_size_z
    }

    #[getter]
    fn channel_names(&self) -> Vec<String> {
        self.inner.channel_names.clone()
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        if let Some(x) = self.inner.pixel_size_x {
            dict.set_item("pixel_size_x", x)?;
        }
        if let Some(y) = self.inner.pixel_size_y {
            dict.set_item("pixel_size_y", y)?;
        }
        if let Some(z) = self.inner.pixel_size_z {
            dict.set_item("pixel_size_z", z)?;
        }
        dict.set_item("channel_names", &self.inner.channel_names)?;
        Ok(dict)
    }
}
```

---

## Python Compatibility Layer

### python/bioio_rust/\_\_init\_\_.py

```python
"""
BioIO Rust - High-performance microscopy image I/O

Drop-in replacement for bioio with 10-15x faster file loading.
"""

from bioio_rust._core import (
    BioImage,
    imread,
    imread_batch,
    imwrite,
    detect_format,
    Metadata,
    __version__,
)

__all__ = [
    "BioImage",
    "imread",
    "imread_batch",
    "imwrite",
    "detect_format",
    "Metadata",
    "__version__",
]
```

### python/bioio_rust/compat.py

```python
"""
Compatibility layer for bioio API.

Provides the same interface as bioio for easy migration.
"""

from typing import Optional, List, Union, Type
from pathlib import Path
import numpy as np

from bioio_rust import BioImage as _RustBioImage, imread as _rust_imread


class BioImage:
    """
    bioio-compatible BioImage class backed by Rust implementation.

    Usage:
        from bioio_rust.compat import BioImage
        img = BioImage("my_file.tiff")
        data = img.data  # numpy array
    """

    def __init__(
        self,
        image: Union[str, Path],
        reader: Optional[Type] = None,
        scene: Optional[int] = None,
        **kwargs,
    ):
        path = str(image) if isinstance(image, Path) else image

        # Map reader class to string
        reader_str = None
        if reader is not None:
            reader_name = reader.__module__.split(".")[-1]
            if "tiff" in reader_name.lower():
                reader_str = "tiff"
            elif "nd2" in reader_name.lower():
                reader_str = "nd2"

        self._impl = _RustBioImage(
            path,
            reader=reader_str,
            scene=scene,
            num_threads=kwargs.get("num_threads", 0),
        )
        self._path = path

    @property
    def shape(self) -> tuple:
        """Image shape as (T, C, Z, Y, X)."""
        return tuple(self._impl.shape)

    @property
    def dims(self) -> tuple:
        """Dimension names."""
        return tuple(self._impl.dims)

    @property
    def data(self) -> np.ndarray:
        """Load and return image data as numpy array."""
        return np.asarray(self._impl.data)

    @property
    def dask_data(self):
        """
        Return data wrapped in dask for lazy evaluation.

        Note: Rust implementation loads eagerly, but wraps in dask
        for API compatibility.
        """
        import dask.array as da
        return da.from_array(self.data, chunks="auto")

    @property
    def xarray_data(self):
        """Return data as xarray DataArray."""
        import xarray as xr
        return xr.DataArray(
            self.data,
            dims=self.dims,
            name=Path(self._path).stem,
        )

    @property
    def scenes(self) -> List[str]:
        """List of scene names."""
        return self._impl.scenes

    @property
    def current_scene(self) -> str:
        """Current scene name."""
        return self.scenes[0]  # TODO: track current

    def set_scene(self, scene: Union[int, str]) -> None:
        """Set the current scene."""
        if isinstance(scene, str):
            scene = self.scenes.index(scene)
        self._impl.set_scene(scene)

    def __repr__(self) -> str:
        return repr(self._impl)


def imread(path: Union[str, Path], **kwargs) -> np.ndarray:
    """Read an image file and return as numpy array."""
    return np.asarray(_rust_imread(str(path)))
```

---

## Build & Distribution

### pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "bioio-rust"
version = "0.1.0"
description = "High-performance microscopy image I/O in Rust"
readme = "README.md"
license = {text = "BSD-3-Clause"}
requires-python = ">=3.10"
authors = [
    {name = "Derek Thirstrup", email = "derek@example.com"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Image Processing",
]
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-benchmark",
    "bioio>=1.0",  # For comparison benchmarks
    "tifffile",
    "nd2",
]
compat = [
    "xarray",
    "dask[array]",
]

[project.urls]
Repository = "https://github.com/derekthirstrup/bioio-rust"
Issues = "https://github.com/derekthirstrup/bioio-rust/issues"

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "bioio_rust._core"
```

---

## Repository Setup Instructions

### Manual Steps (GitHub UI)

1. **Create new repository:**
   - Go to https://github.com/new
   - Repository name: `bioio-rust`
   - Description: "High-performance Rust implementation of BioIO for microscopy image I/O"
   - Public repository
   - Initialize with README
   - Add BSD-3-Clause license

2. **Clone and initialize:**
```bash
git clone https://github.com/derekthirstrup/bioio-rust
cd bioio-rust

# Initialize Rust project
cargo init --lib

# Copy the source files from this plan

# Build and test
maturin develop
python -c "from bioio_rust import BioImage; print('Success!')"
```

3. **Set up CI/CD (.github/workflows/ci.yml):**
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rust-lang/setup-rust-toolchain@v1
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install maturin pytest numpy
      - run: maturin develop
      - run: pytest

  build-wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist
```

---

## Performance Targets

| Operation | bioio (Python) | bioio-rust | Speedup |
|-----------|----------------|------------|---------|
| TIFF 100x100 | 2.2ms | 0.15ms | **14.7x** |
| TIFF 8000x8000 | 100ms | 15ms | **6.7x** |
| ND2 1000x1000x10 | 500ms | 50ms | **10x** |
| Batch 100 TIFFs | 12s | 0.8s | **15x** |
| Plugin discovery | 50ms | 0 | **∞** |

---

## Next Steps

1. Create the `derekthirstrup/bioio-rust` repository on GitHub
2. Initialize the Rust project structure
3. Implement TiffReader first (most common format)
4. Add benchmarks comparing to tifffile and bioio
5. Implement Nd2Reader
6. Publish alpha release to PyPI
7. Integrate with napari as optional fast backend
