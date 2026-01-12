//! Image writers
//!
//! This module provides writers for saving microscopy images
//! to various formats.

use crate::error::Result;
use crate::format::ImageFormat;
use crate::metadata::Metadata;
use crate::dimensions::Dimensions;
use ndarray::ArrayD;
use std::path::Path;

pub mod tiff;

/// Writer options
#[derive(Debug, Clone, Default)]
pub struct WriterOptions {
    /// Compression type
    pub compression: Option<String>,
    /// Compression level (0-9)
    pub compression_level: Option<u32>,
    /// Tile size for tiled TIFFs
    pub tile_size: Option<(usize, usize)>,
    /// Use BigTIFF format
    pub bigtiff: bool,
    /// Include OME-XML metadata
    pub ome_xml: bool,
}

impl WriterOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_compression(mut self, compression: &str) -> Self {
        self.compression = Some(compression.to_string());
        self
    }

    pub fn with_bigtiff(mut self, bigtiff: bool) -> Self {
        self.bigtiff = bigtiff;
        self
    }
}

/// Core writer trait
pub trait Writer {
    /// Write image data to file
    fn write(
        path: &Path,
        data: &ArrayD<u8>,
        dimensions: &Dimensions,
        metadata: Option<&Metadata>,
        options: &WriterOptions,
    ) -> Result<()>;

    /// Get the format this writer produces
    fn format() -> ImageFormat;

    /// Get supported file extensions
    fn extensions() -> &'static [&'static str];
}

/// Write an image to file with automatic format detection
pub fn imwrite(
    path: &Path,
    data: &ArrayD<u8>,
    dimensions: &Dimensions,
    metadata: Option<&Metadata>,
    options: Option<&WriterOptions>,
) -> Result<()> {
    let opts = options.cloned().unwrap_or_default();

    // Determine format from extension
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "tif" | "tiff" => tiff::TiffWriter::write(path, data, dimensions, metadata, &opts),
        _ => Err(crate::BioIoError::UnsupportedFormat(format!(
            "No writer for extension: {}",
            ext
        ))),
    }
}
