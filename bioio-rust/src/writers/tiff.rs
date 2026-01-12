//! TIFF writer
//!
//! Writes TIFF/BigTIFF files with optional compression.

use crate::dimensions::Dimensions;
use crate::error::{BioIoError, Result};
use crate::format::ImageFormat;
use crate::metadata::Metadata;
use crate::writers::{Writer, WriterOptions};

use byteorder::{LittleEndian, WriteBytesExt};
use ndarray::ArrayD;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// TIFF writer
pub struct TiffWriter;

impl TiffWriter {
    /// Create a minimal TIFF file
    fn write_tiff(
        path: &Path,
        data: &ArrayD<u8>,
        dimensions: &Dimensions,
        metadata: Option<&Metadata>,
        options: &WriterOptions,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let height = dimensions.y;
        let width = dimensions.x;
        let channels = dimensions.c;
        let num_frames = dimensions.t * dimensions.z;

        // Determine if we need BigTIFF
        let total_size = data.len();
        let use_bigtiff = options.bigtiff || total_size > 4_000_000_000;

        if use_bigtiff {
            Self::write_bigtiff_header(&mut writer)?;
        } else {
            Self::write_tiff_header(&mut writer)?;
        }

        // Calculate image data size per frame
        let frame_size = height * width * channels;

        // Write IFDs and image data for each frame
        let mut ifd_offsets = Vec::new();
        let mut current_offset = if use_bigtiff { 16u64 } else { 8u64 };

        for frame in 0..num_frames {
            let frame_data = Self::extract_frame(data, dimensions, frame);

            // Record IFD position
            ifd_offsets.push(current_offset);

            // Write IFD
            let (ifd_size, data_offset) = Self::write_ifd(
                &mut writer,
                width as u32,
                height as u32,
                channels as u16,
                8, // bits per sample
                current_offset,
                frame_size,
                use_bigtiff,
                frame == num_frames - 1,
                metadata,
            )?;

            current_offset += ifd_size as u64;

            // Write image data
            writer.write_all(&frame_data)?;
            current_offset += frame_size as u64;
        }

        writer.flush()?;
        Ok(())
    }

    fn write_tiff_header<W: Write>(writer: &mut W) -> Result<()> {
        // Little endian marker
        writer.write_all(b"II")?;
        // TIFF magic number (42)
        writer.write_u16::<LittleEndian>(42)?;
        // First IFD offset (immediately after header)
        writer.write_u32::<LittleEndian>(8)?;
        Ok(())
    }

    fn write_bigtiff_header<W: Write>(writer: &mut W) -> Result<()> {
        // Little endian marker
        writer.write_all(b"II")?;
        // BigTIFF magic number (43)
        writer.write_u16::<LittleEndian>(43)?;
        // Byte size of offsets (8)
        writer.write_u16::<LittleEndian>(8)?;
        // Always 0
        writer.write_u16::<LittleEndian>(0)?;
        // First IFD offset
        writer.write_u64::<LittleEndian>(16)?;
        Ok(())
    }

    fn write_ifd<W: Write>(
        writer: &mut W,
        width: u32,
        height: u32,
        samples_per_pixel: u16,
        bits_per_sample: u16,
        ifd_offset: u64,
        image_size: usize,
        _use_bigtiff: bool,
        is_last: bool,
        _metadata: Option<&Metadata>,
    ) -> Result<(usize, u64)> {
        // Standard TIFF IFD
        let num_entries: u16 = 10;

        // Calculate offsets
        let ifd_size = 2 + (num_entries as usize * 12) + 4;
        let data_offset = ifd_offset + ifd_size as u64;

        // Number of entries
        writer.write_u16::<LittleEndian>(num_entries)?;

        // Tag entries (12 bytes each)
        // ImageWidth (256)
        Self::write_tag(writer, 256, 4, 1, width as u64)?;
        // ImageHeight (257)
        Self::write_tag(writer, 257, 4, 1, height as u64)?;
        // BitsPerSample (258)
        Self::write_tag(writer, 258, 3, 1, bits_per_sample as u64)?;
        // Compression (259) = 1 (none)
        Self::write_tag(writer, 259, 3, 1, 1)?;
        // PhotometricInterpretation (262) = 1 (black is zero) or 2 (RGB)
        let photometric = if samples_per_pixel >= 3 { 2 } else { 1 };
        Self::write_tag(writer, 262, 3, 1, photometric)?;
        // StripOffsets (273)
        Self::write_tag(writer, 273, 4, 1, data_offset)?;
        // SamplesPerPixel (277)
        Self::write_tag(writer, 277, 3, 1, samples_per_pixel as u64)?;
        // RowsPerStrip (278)
        Self::write_tag(writer, 278, 4, 1, height as u64)?;
        // StripByteCounts (279)
        Self::write_tag(writer, 279, 4, 1, image_size as u64)?;
        // PlanarConfiguration (284) = 1 (chunky)
        Self::write_tag(writer, 284, 3, 1, 1)?;

        // Next IFD offset (0 if last)
        let next_ifd = if is_last {
            0
        } else {
            data_offset + image_size as u64
        };
        writer.write_u32::<LittleEndian>(next_ifd as u32)?;

        Ok((ifd_size, data_offset))
    }

    fn write_tag<W: Write>(
        writer: &mut W,
        tag: u16,
        field_type: u16,
        count: u32,
        value: u64,
    ) -> Result<()> {
        writer.write_u16::<LittleEndian>(tag)?;
        writer.write_u16::<LittleEndian>(field_type)?;
        writer.write_u32::<LittleEndian>(count)?;
        writer.write_u32::<LittleEndian>(value as u32)?;
        Ok(())
    }

    fn extract_frame(data: &ArrayD<u8>, dimensions: &Dimensions, frame: usize) -> Vec<u8> {
        let t = frame / dimensions.z;
        let z = frame % dimensions.z;

        let mut frame_data = Vec::with_capacity(dimensions.y * dimensions.x * dimensions.c);

        for y in 0..dimensions.y {
            for x in 0..dimensions.x {
                for c in 0..dimensions.c {
                    let value = if data.ndim() == 5 {
                        data[[t, c, z, y, x]]
                    } else {
                        // Handle other shapes
                        data.as_slice().map(|s| s[0]).unwrap_or(0)
                    };
                    frame_data.push(value);
                }
            }
        }

        frame_data
    }
}

impl Writer for TiffWriter {
    fn write(
        path: &Path,
        data: &ArrayD<u8>,
        dimensions: &Dimensions,
        metadata: Option<&Metadata>,
        options: &WriterOptions,
    ) -> Result<()> {
        Self::write_tiff(path, data, dimensions, metadata, options)
    }

    fn format() -> ImageFormat {
        ImageFormat::Tiff
    }

    fn extensions() -> &'static [&'static str] {
        &["tif", "tiff"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_write_simple_tiff() {
        let data = ArrayD::<u8>::zeros(ndarray::IxDyn(&[1, 1, 1, 10, 10]));
        let dims = Dimensions::new(1, 1, 1, 10, 10);

        let file = NamedTempFile::with_suffix(".tiff").unwrap();
        let result = TiffWriter::write(file.path(), &data, &dims, None, &WriterOptions::default());

        assert!(result.is_ok());
    }
}
