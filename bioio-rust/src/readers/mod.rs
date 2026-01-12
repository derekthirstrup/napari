//! Image format readers
//!
//! This module provides the core `Reader` trait and implementations
//! for various microscopy image formats.

use crate::error::Result;
use crate::format::ImageFormat;
use crate::metadata::Metadata;
use crate::dimensions::Dimensions;
use ndarray::ArrayD;
use std::path::Path;

pub mod tiff;
pub mod ome_tiff;
pub mod nd2;
pub mod png;

/// Data type for image pixels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// Unsigned 8-bit integer
    U8,
    /// Unsigned 16-bit integer
    U16,
    /// Unsigned 32-bit integer
    U32,
    /// Signed 8-bit integer
    I8,
    /// Signed 16-bit integer
    I16,
    /// Signed 32-bit integer
    I32,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
}

impl DType {
    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::F64 => 8,
        }
    }

    /// NumPy dtype string
    pub fn numpy_str(&self) -> &'static str {
        match self {
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::F32 => "float32",
            DType::F64 => "float64",
        }
    }

    /// From bits per sample and sample format
    pub fn from_bits_and_format(bits: u16, signed: bool, float: bool) -> Self {
        match (bits, signed, float) {
            (8, false, false) => DType::U8,
            (8, true, false) => DType::I8,
            (16, false, false) => DType::U16,
            (16, true, false) => DType::I16,
            (32, false, false) => DType::U32,
            (32, true, false) => DType::I32,
            (32, _, true) => DType::F32,
            (64, _, true) => DType::F64,
            _ => DType::U8, // Default
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.numpy_str())
    }
}

/// Region specification for partial reads
#[derive(Debug, Clone, Default)]
pub struct Region {
    /// Time range
    pub t: Option<std::ops::Range<usize>>,
    /// Channel range
    pub c: Option<std::ops::Range<usize>>,
    /// Z range
    pub z: Option<std::ops::Range<usize>>,
    /// Y range (rows)
    pub y: Option<std::ops::Range<usize>>,
    /// X range (columns)
    pub x: Option<std::ops::Range<usize>>,
}

impl Region {
    /// Create a new empty region (full image)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a region with all dimensions specified
    pub fn full(dims: &Dimensions) -> Self {
        Self {
            t: Some(0..dims.t),
            c: Some(0..dims.c),
            z: Some(0..dims.z),
            y: Some(0..dims.y),
            x: Some(0..dims.x),
        }
    }

    /// Set time range
    pub fn with_t(mut self, range: std::ops::Range<usize>) -> Self {
        self.t = Some(range);
        self
    }

    /// Set channel range
    pub fn with_c(mut self, range: std::ops::Range<usize>) -> Self {
        self.c = Some(range);
        self
    }

    /// Set Z range
    pub fn with_z(mut self, range: std::ops::Range<usize>) -> Self {
        self.z = Some(range);
        self
    }

    /// Set Y range
    pub fn with_y(mut self, range: std::ops::Range<usize>) -> Self {
        self.y = Some(range);
        self
    }

    /// Set X range
    pub fn with_x(mut self, range: std::ops::Range<usize>) -> Self {
        self.x = Some(range);
        self
    }

    /// Get the shape of this region
    pub fn shape(&self, dims: &Dimensions) -> [usize; 5] {
        [
            self.t.as_ref().map(|r| r.len()).unwrap_or(dims.t),
            self.c.as_ref().map(|r| r.len()).unwrap_or(dims.c),
            self.z.as_ref().map(|r| r.len()).unwrap_or(dims.z),
            self.y.as_ref().map(|r| r.len()).unwrap_or(dims.y),
            self.x.as_ref().map(|r| r.len()).unwrap_or(dims.x),
        ]
    }
}

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
    /// Enable memory mapping
    pub use_mmap: bool,
    /// Chunk size for dask-like lazy loading
    pub chunk_size: Option<Vec<usize>>,
}

impl ReaderOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_scene(mut self, scene: usize) -> Self {
        self.scene = Some(scene);
        self
    }

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }
}

/// Core reader trait implemented by all format readers
///
/// This trait provides a unified interface for reading microscopy images
/// across different file formats. Implementations must be thread-safe
/// (`Send + Sync`) to enable parallel reading.
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
        &self.dimensions().shape()
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

    /// Get current scene index
    fn current_scene(&self) -> usize;

    /// Read all data into memory as u8 array
    ///
    /// For higher bit-depth images, the data is stored as raw bytes.
    /// Use `read_all_typed` for type-safe access.
    fn read_all(&self) -> Result<ArrayD<u8>>;

    /// Read a specific region/slice
    fn read_region(&self, region: &Region) -> Result<ArrayD<u8>>;

    /// Read with parallel decoding using specified number of threads
    ///
    /// If `num_threads` is 0, uses all available cores.
    fn read_parallel(&self, num_threads: usize) -> Result<ArrayD<u8>>;

    /// Get the file path
    fn path(&self) -> &str;
}

/// Extension trait for typed array reading
pub trait ReaderExt: Reader {
    /// Read all data as u16 array
    fn read_all_u16(&self) -> Result<ArrayD<u16>> {
        let data = self.read_all()?;
        let shape = data.shape().to_vec();
        let new_shape: Vec<usize> = shape.iter().take(shape.len() - 1).copied().collect();

        // Reinterpret bytes as u16
        let bytes = data.into_raw_vec();
        let u16_data: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        Ok(ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), u16_data)
            .map_err(|e| crate::BioIoError::Other(e.to_string()))?)
    }

    /// Read all data as f32 array
    fn read_all_f32(&self) -> Result<ArrayD<f32>> {
        let data = self.read_all()?;
        let shape = data.shape().to_vec();
        let new_shape: Vec<usize> = shape.iter().take(shape.len() - 1).copied().collect();

        let bytes = data.into_raw_vec();
        let f32_data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(ArrayD::from_shape_vec(ndarray::IxDyn(&new_shape), f32_data)
            .map_err(|e| crate::BioIoError::Other(e.to_string()))?)
    }
}

impl<T: Reader> ReaderExt for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::U16.size_bytes(), 2);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_region_shape() {
        let dims = Dimensions::new(2, 3, 4, 100, 200);
        let region = Region::new()
            .with_t(0..1)
            .with_z(1..3);

        let shape = region.shape(&dims);
        assert_eq!(shape, [1, 3, 2, 100, 200]);
    }
}
