//! Image format detection via magic bytes
//!
//! This module provides fast format detection by reading only the first
//! few bytes of a file, avoiding the overhead of iterating through plugins.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::error::{BioIoError, Result};

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    /// Standard TIFF
    Tiff,
    /// BigTIFF (>4GB files)
    BigTiff,
    /// OME-TIFF with XML metadata
    OmeTiff,
    /// Nikon ND2
    Nd2,
    /// Zeiss CZI
    Czi,
    /// Leica LIF
    Lif,
    /// Zarr directory
    Zarr,
    /// PNG image
    Png,
    /// JPEG image
    Jpeg,
    /// NumPy array file
    Npy,
    /// Unknown format
    Unknown,
}

impl ImageFormat {
    /// Detect format from magic bytes (fast, no iteration)
    ///
    /// This reads only the first 16 bytes of the file to determine format,
    /// avoiding the overhead of loading plugins and testing each one.
    pub fn detect(path: &Path) -> Result<Self> {
        // Check for Zarr directory first
        if path.is_dir() {
            let zarray = path.join(".zarray");
            let zgroup = path.join(".zgroup");
            if zarray.exists() || zgroup.exists() {
                return Ok(ImageFormat::Zarr);
            }
        }

        let mut file = File::open(path)?;
        let mut magic = [0u8; 16];
        let bytes_read = file.read(&mut magic)?;

        if bytes_read < 4 {
            return Err(BioIoError::Other("File too small to detect format".into()));
        }

        Ok(Self::from_magic(&magic, path))
    }

    /// Detect format from magic bytes and optional path hint
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
            if let Some(name) = path.file_name() {
                let name = name.to_string_lossy().to_lowercase();
                if name.ends_with(".ome.tiff") || name.ends_with(".ome.tif") {
                    return ImageFormat::OmeTiff;
                }
            }

            return ImageFormat::Tiff;
        }

        // ND2: Starts with 0xDADADADA (little-endian magic)
        if magic.len() >= 4 {
            let nd2_magic = u32::from_le_bytes([magic[0], magic[1], magic[2], magic[3]]);
            if nd2_magic == 0xDADADADA {
                return ImageFormat::Nd2;
            }
        }

        // CZI: Starts with "ZISRAWFILE"
        if magic.starts_with(b"ZISRAWFILE") {
            return ImageFormat::Czi;
        }

        // LIF: Starts with 0x70 (memory check value)
        if magic.len() >= 4 && magic[0] == 0x70 {
            // Additional check for LIF format
            let test = i32::from_le_bytes([magic[0], magic[1], magic[2], magic[3]]);
            if test == 0x70 {
                return ImageFormat::Lif;
            }
        }

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if magic.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            return ImageFormat::Png;
        }

        // JPEG: FF D8 FF
        if magic.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return ImageFormat::Jpeg;
        }

        // NumPy: \x93NUMPY
        if magic.starts_with(&[0x93, b'N', b'U', b'M', b'P', b'Y']) {
            return ImageFormat::Npy;
        }

        // Check extension as fallback
        if let Some(ext) = path.extension() {
            match ext.to_string_lossy().to_lowercase().as_str() {
                "tif" | "tiff" => return ImageFormat::Tiff,
                "nd2" => return ImageFormat::Nd2,
                "czi" => return ImageFormat::Czi,
                "lif" => return ImageFormat::Lif,
                "png" => return ImageFormat::Png,
                "jpg" | "jpeg" => return ImageFormat::Jpeg,
                "npy" => return ImageFormat::Npy,
                "zarr" => return ImageFormat::Zarr,
                _ => {}
            }
        }

        ImageFormat::Unknown
    }

    /// Get the canonical file extensions for this format
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            ImageFormat::Tiff => &["tif", "tiff"],
            ImageFormat::BigTiff => &["tif", "tiff", "btf"],
            ImageFormat::OmeTiff => &["ome.tif", "ome.tiff"],
            ImageFormat::Nd2 => &["nd2"],
            ImageFormat::Czi => &["czi"],
            ImageFormat::Lif => &["lif"],
            ImageFormat::Zarr => &["zarr"],
            ImageFormat::Png => &["png"],
            ImageFormat::Jpeg => &["jpg", "jpeg"],
            ImageFormat::Npy => &["npy"],
            ImageFormat::Unknown => &[],
        }
    }

    /// Get human-readable format name
    pub fn name(&self) -> &'static str {
        match self {
            ImageFormat::Tiff => "TIFF",
            ImageFormat::BigTiff => "BigTIFF",
            ImageFormat::OmeTiff => "OME-TIFF",
            ImageFormat::Nd2 => "Nikon ND2",
            ImageFormat::Czi => "Zeiss CZI",
            ImageFormat::Lif => "Leica LIF",
            ImageFormat::Zarr => "Zarr",
            ImageFormat::Png => "PNG",
            ImageFormat::Jpeg => "JPEG",
            ImageFormat::Npy => "NumPy",
            ImageFormat::Unknown => "Unknown",
        }
    }

    /// Check if format is currently supported for reading
    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            ImageFormat::Tiff
                | ImageFormat::BigTiff
                | ImageFormat::OmeTiff
                | ImageFormat::Nd2
                | ImageFormat::Png
        )
    }

    /// Get all supported formats
    pub fn supported_formats() -> Vec<ImageFormat> {
        vec![
            ImageFormat::Tiff,
            ImageFormat::BigTiff,
            ImageFormat::OmeTiff,
            ImageFormat::Nd2,
            ImageFormat::Png,
        ]
    }
}

impl std::fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_detect_tiff_little_endian() {
        let mut file = NamedTempFile::with_suffix(".tiff").unwrap();
        // Little-endian TIFF header
        file.write_all(&[b'I', b'I', 42, 0, 8, 0, 0, 0]).unwrap();
        file.flush().unwrap();

        let format = ImageFormat::detect(file.path()).unwrap();
        assert_eq!(format, ImageFormat::Tiff);
    }

    #[test]
    fn test_detect_tiff_big_endian() {
        let mut file = NamedTempFile::with_suffix(".tiff").unwrap();
        // Big-endian TIFF header
        file.write_all(&[b'M', b'M', 0, 42, 0, 0, 0, 8]).unwrap();
        file.flush().unwrap();

        let format = ImageFormat::detect(file.path()).unwrap();
        assert_eq!(format, ImageFormat::Tiff);
    }

    #[test]
    fn test_detect_bigtiff() {
        let mut file = NamedTempFile::with_suffix(".tiff").unwrap();
        // BigTIFF header (version 43)
        file.write_all(&[b'I', b'I', 43, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        file.flush().unwrap();

        let format = ImageFormat::detect(file.path()).unwrap();
        assert_eq!(format, ImageFormat::BigTiff);
    }

    #[test]
    fn test_detect_png() {
        let mut file = NamedTempFile::with_suffix(".png").unwrap();
        // PNG header
        file.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
            .unwrap();
        file.flush().unwrap();

        let format = ImageFormat::detect(file.path()).unwrap();
        assert_eq!(format, ImageFormat::Png);
    }

    #[test]
    fn test_detect_nd2() {
        let mut file = NamedTempFile::with_suffix(".nd2").unwrap();
        // ND2 magic bytes
        file.write_all(&[0xDA, 0xDA, 0xDA, 0xDA, 0, 0, 0, 0]).unwrap();
        file.flush().unwrap();

        let format = ImageFormat::detect(file.path()).unwrap();
        assert_eq!(format, ImageFormat::Nd2);
    }
}
