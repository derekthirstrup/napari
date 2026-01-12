//! # BioIO Rust
//!
//! High-performance microscopy image I/O library with Python bindings.
//!
//! This crate provides fast, parallel image reading for microscopy formats
//! with zero-copy Python/NumPy integration via PyO3.
//!
//! ## Supported Formats
//!
//! - TIFF/BigTIFF (via native implementation)
//! - OME-TIFF (with XML metadata parsing)
//! - ND2 (Nikon NIS-Elements)
//! - PNG, JPEG (basic support)
//!
//! ## Example
//!
//! ```rust,ignore
//! use bioio_rust::{BioImage, ImageFormat};
//!
//! let img = BioImage::open("image.tiff")?;
//! println!("Shape: {:?}", img.shape());
//! let data = img.read_all()?;
//! ```
//!
//! ## Python Usage
//!
//! ```python
//! from bioio_rust import BioImage, imread
//!
//! # Class-based API
//! img = BioImage("image.tiff")
//! data = img.data  # numpy array
//!
//! # Function API
//! data = imread("image.tiff")
//!
//! # Batch parallel reading
//! from bioio_rust import imread_batch
//! arrays = imread_batch(["img1.tiff", "img2.tiff"], num_threads=8)
//! ```

use pyo3::prelude::*;

pub mod error;
pub mod format;
pub mod dimensions;
pub mod metadata;
pub mod readers;
pub mod writers;
pub mod parallel;
mod python;

// Re-exports for public API
pub use error::{BioIoError, Result};
pub use format::ImageFormat;
pub use dimensions::{Dimensions, DimensionOrder};
pub use metadata::Metadata;
pub use readers::{Reader, ReaderOptions, DType, Region};

/// Python module definition
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging
    let _ = env_logger::try_init();

    // Core classes
    m.add_class::<python::PyBioImage>()?;
    m.add_class::<python::PyMetadata>()?;
    m.add_class::<python::PyDimensions>()?;

    // Fast functions for direct use
    m.add_function(wrap_pyfunction!(python::imread, m)?)?;
    m.add_function(wrap_pyfunction!(python::imread_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python::imwrite, m)?)?;
    m.add_function(wrap_pyfunction!(python::detect_format, m)?)?;
    m.add_function(wrap_pyfunction!(python::get_supported_formats, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// High-level BioImage struct for Rust usage
pub struct BioImage {
    reader: Box<dyn Reader>,
    path: String,
}

impl BioImage {
    /// Open an image file with automatic format detection
    pub fn open(path: &str) -> Result<Self> {
        Self::open_with_options(path, &ReaderOptions::default())
    }

    /// Open an image file with specific options
    pub fn open_with_options(path: &str, options: &ReaderOptions) -> Result<Self> {
        use std::path::Path;

        let path_obj = Path::new(path);
        let format = ImageFormat::detect(path_obj)?;

        let reader: Box<dyn Reader> = match format {
            ImageFormat::Tiff | ImageFormat::BigTiff => {
                Box::new(readers::tiff::TiffReader::open(path_obj, options)?)
            }
            ImageFormat::OmeTiff => {
                Box::new(readers::ome_tiff::OmeTiffReader::open(path_obj, options)?)
            }
            ImageFormat::Nd2 => {
                Box::new(readers::nd2::Nd2Reader::open(path_obj, options)?)
            }
            ImageFormat::Png => {
                Box::new(readers::png::PngReader::open(path_obj, options)?)
            }
            _ => {
                return Err(BioIoError::UnsupportedFormat(format!("{:?}", format)));
            }
        };

        Ok(Self {
            reader,
            path: path.to_string(),
        })
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> &Dimensions {
        self.reader.dimensions()
    }

    /// Get image shape as slice
    pub fn shape(&self) -> Vec<usize> {
        self.dimensions().shape().to_vec()
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.reader.dtype()
    }

    /// Get metadata
    pub fn metadata(&self) -> &Metadata {
        self.reader.metadata()
    }

    /// Read all data
    pub fn read_all(&self) -> Result<ndarray::ArrayD<u8>> {
        self.reader.read_all()
    }

    /// Read with parallel decoding
    pub fn read_parallel(&self, num_threads: usize) -> Result<ndarray::ArrayD<u8>> {
        self.reader.read_parallel(num_threads)
    }

    /// Get the file path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the image format
    pub fn format(&self) -> ImageFormat {
        self.reader.format()
    }
}
