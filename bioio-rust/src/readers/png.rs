//! PNG image reader
//!
//! Basic PNG support for completeness. PNG files are typically
//! not used for serious microscopy work but are useful for
//! screenshots, documentation images, etc.

use crate::dimensions::Dimensions;
use crate::error::{BioIoError, Result};
use crate::format::ImageFormat;
use crate::metadata::Metadata;
use crate::readers::{DType, Reader, ReaderOptions, Region};

use ndarray::{ArrayD, IxDyn};
use png::ColorType;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// PNG reader
pub struct PngReader {
    /// File path
    path: String,
    /// Image dimensions
    dimensions: Dimensions,
    /// Data type
    dtype: DType,
    /// Metadata
    metadata: Metadata,
    /// Decoded image data (PNG is not seekable, so we decode once)
    data: Vec<u8>,
    /// Color type
    color_type: ColorType,
    /// Bit depth
    bit_depth: u8,
}

impl PngReader {
    /// Get number of channels from color type
    fn channels_from_color_type(color_type: ColorType) -> usize {
        match color_type {
            ColorType::Grayscale => 1,
            ColorType::GrayscaleAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
            ColorType::Indexed => 1, // Palette-based
        }
    }
}

impl Reader for PngReader {
    fn open(path: &Path, _options: &ReaderOptions) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let decoder = png::Decoder::new(reader);
        let mut png_reader = decoder
            .read_info()
            .map_err(|e| BioIoError::InvalidPng(e.to_string()))?;

        let info = png_reader.info();
        let width = info.width as usize;
        let height = info.height as usize;
        let color_type = info.color_type;
        let bit_depth = info.bit_depth as u8;

        let channels = Self::channels_from_color_type(color_type);

        // Allocate buffer and decode
        let mut data = vec![0u8; png_reader.output_buffer_size()];
        let output_info = png_reader
            .next_frame(&mut data)
            .map_err(|e| BioIoError::InvalidPng(e.to_string()))?;

        data.truncate(output_info.buffer_size());

        // Build dimensions (single 2D image with channels)
        let dimensions = Dimensions::new(1, channels, 1, height, width);

        // Determine dtype
        let dtype = match bit_depth {
            8 => DType::U8,
            16 => DType::U16,
            _ => DType::U8,
        };

        let metadata = Metadata::new();

        Ok(Self {
            path: path.to_string_lossy().into_owned(),
            dimensions,
            dtype,
            metadata,
            data,
            color_type,
            bit_depth,
        })
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Png
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
        1
    }

    fn scene_names(&self) -> Vec<String> {
        vec!["Scene_0".to_string()]
    }

    fn set_scene(&mut self, scene: usize) -> Result<()> {
        if scene != 0 {
            return Err(BioIoError::SceneNotFound(scene, 1));
        }
        Ok(())
    }

    fn current_scene(&self) -> usize {
        0
    }

    fn read_all(&self) -> Result<ArrayD<u8>> {
        let dims = &self.dimensions;
        let bytes_per_sample = if self.bit_depth == 16 { 2 } else { 1 };

        // Create output array in TCZYX order
        let mut output = ArrayD::<u8>::zeros(IxDyn(&[dims.t, dims.c, dims.z, dims.y, dims.x]));

        // PNG stores as interleaved RGBRGBRGB... per row
        let row_bytes = dims.x * dims.c * bytes_per_sample;

        for y in 0..dims.y {
            for x in 0..dims.x {
                for c in 0..dims.c {
                    let src_idx = y * row_bytes + (x * dims.c + c) * bytes_per_sample;

                    if src_idx < self.data.len() {
                        if bytes_per_sample == 1 {
                            output[[0, c, 0, y, x]] = self.data[src_idx];
                        } else {
                            // For 16-bit, use high byte for u8 output
                            output[[0, c, 0, y, x]] = self.data[src_idx + 1];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    fn read_region(&self, region: &Region) -> Result<ArrayD<u8>> {
        let full = self.read_all()?;

        let dims = self.dimensions();
        let t_range = region.t.clone().unwrap_or(0..dims.t);
        let c_range = region.c.clone().unwrap_or(0..dims.c);
        let z_range = region.z.clone().unwrap_or(0..dims.z);
        let y_range = region.y.clone().unwrap_or(0..dims.y);
        let x_range = region.x.clone().unwrap_or(0..dims.x);

        let shape = region.shape(dims);
        let mut output = ArrayD::<u8>::zeros(IxDyn(&shape));

        for (out_t, t) in t_range.enumerate() {
            for (out_c, c) in c_range.clone().enumerate() {
                for (out_z, z) in z_range.clone().enumerate() {
                    for (out_y, y) in y_range.clone().enumerate() {
                        for (out_x, x) in x_range.clone().enumerate() {
                            output[[out_t, out_c, out_z, out_y, out_x]] = full[[t, c, z, y, x]];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    fn read_parallel(&self, _num_threads: usize) -> Result<ArrayD<u8>> {
        // PNG doesn't benefit much from parallel reading (already decoded)
        self.read_all()
    }

    fn path(&self) -> &str {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channels_from_color_type() {
        assert_eq!(PngReader::channels_from_color_type(ColorType::Grayscale), 1);
        assert_eq!(PngReader::channels_from_color_type(ColorType::Rgb), 3);
        assert_eq!(PngReader::channels_from_color_type(ColorType::Rgba), 4);
    }
}
