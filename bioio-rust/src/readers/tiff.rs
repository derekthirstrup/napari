//! TIFF/BigTIFF reader with parallel decoding
//!
//! This module provides a high-performance TIFF reader that supports:
//! - Standard TIFF and BigTIFF (>4GB files)
//! - Various compression methods (None, LZW, Deflate, JPEG, Zstd)
//! - Stripped and tiled TIFFs
//! - Parallel strip/tile decompression via rayon
//! - Memory-mapped file access for zero-copy reading

use crate::dimensions::Dimensions;
use crate::error::{BioIoError, Result};
use crate::format::ImageFormat;
use crate::metadata::{Metadata, SizeUnit};
use crate::readers::{DType, Reader, ReaderOptions, Region};

use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use ndarray::{Array3, ArrayD, IxDyn};
use rayon::prelude::*;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

/// TIFF compression types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None = 1,
    Lzw = 5,
    Jpeg = 7,
    Deflate = 8,
    AdobeDeflate = 32946,
    Zstd = 50000,
    Unknown(u16),
}

impl From<u16> for Compression {
    fn from(value: u16) -> Self {
        match value {
            1 => Compression::None,
            5 => Compression::Lzw,
            7 => Compression::Jpeg,
            8 => Compression::Deflate,
            32946 => Compression::AdobeDeflate,
            50000 => Compression::Zstd,
            v => Compression::Unknown(v),
        }
    }
}

/// TIFF tag IDs
#[allow(dead_code)]
mod tags {
    pub const IMAGE_WIDTH: u16 = 256;
    pub const IMAGE_HEIGHT: u16 = 257;
    pub const BITS_PER_SAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PHOTOMETRIC: u16 = 262;
    pub const IMAGE_DESCRIPTION: u16 = 270;
    pub const STRIP_OFFSETS: u16 = 273;
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    pub const ROWS_PER_STRIP: u16 = 278;
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    pub const X_RESOLUTION: u16 = 282;
    pub const Y_RESOLUTION: u16 = 283;
    pub const PLANAR_CONFIG: u16 = 284;
    pub const RESOLUTION_UNIT: u16 = 296;
    pub const SOFTWARE: u16 = 305;
    pub const DATETIME: u16 = 306;
    pub const TILE_WIDTH: u16 = 322;
    pub const TILE_HEIGHT: u16 = 323;
    pub const TILE_OFFSETS: u16 = 324;
    pub const TILE_BYTE_COUNTS: u16 = 325;
    pub const SAMPLE_FORMAT: u16 = 339;
    pub const IMAGEJ_METADATA: u16 = 50839;
}

/// Parsed IFD (Image File Directory) entry representing one image page
#[derive(Debug, Clone)]
struct IfdEntry {
    /// Image width in pixels
    width: u32,
    /// Image height in pixels
    height: u32,
    /// Bits per sample (8, 16, 32, etc.)
    bits_per_sample: u16,
    /// Samples per pixel (1 for grayscale, 3 for RGB, etc.)
    samples_per_pixel: u16,
    /// Compression type
    compression: Compression,
    /// Photometric interpretation
    photometric: u16,
    /// Sample format (1=uint, 2=int, 3=float)
    sample_format: u16,
    /// Planar configuration (1=chunky, 2=planar)
    planar_config: u16,
    /// Rows per strip (for stripped TIFFs)
    rows_per_strip: u32,
    /// Strip offsets
    strip_offsets: Vec<u64>,
    /// Strip byte counts
    strip_byte_counts: Vec<u64>,
    /// Tile width (0 if not tiled)
    tile_width: u32,
    /// Tile height (0 if not tiled)
    tile_height: u32,
    /// Tile offsets
    tile_offsets: Vec<u64>,
    /// Tile byte counts
    tile_byte_counts: Vec<u64>,
    /// X resolution (pixels per unit)
    x_resolution: Option<f64>,
    /// Y resolution (pixels per unit)
    y_resolution: Option<f64>,
    /// Resolution unit (1=none, 2=inch, 3=cm)
    resolution_unit: u16,
    /// Image description (may contain OME-XML or ImageJ metadata)
    description: Option<String>,
    /// Software used to create the image
    software: Option<String>,
}

impl IfdEntry {
    fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            bits_per_sample: 8,
            samples_per_pixel: 1,
            compression: Compression::None,
            photometric: 1,
            sample_format: 1,
            planar_config: 1,
            rows_per_strip: u32::MAX,
            strip_offsets: Vec::new(),
            strip_byte_counts: Vec::new(),
            tile_width: 0,
            tile_height: 0,
            tile_offsets: Vec::new(),
            tile_byte_counts: Vec::new(),
            x_resolution: None,
            y_resolution: None,
            resolution_unit: 2,
            description: None,
            software: None,
        }
    }

    /// Check if this IFD uses tiled storage
    fn is_tiled(&self) -> bool {
        self.tile_width > 0 && self.tile_height > 0
    }

    /// Get bytes per pixel
    fn bytes_per_pixel(&self) -> usize {
        (self.bits_per_sample as usize / 8) * self.samples_per_pixel as usize
    }

    /// Calculate expected decompressed size for a strip
    fn strip_size(&self, strip_idx: usize) -> usize {
        let strips = self.num_strips();
        let rows = if strip_idx == strips - 1 {
            // Last strip may have fewer rows
            let remaining = self.height as usize % self.rows_per_strip as usize;
            if remaining == 0 {
                self.rows_per_strip as usize
            } else {
                remaining
            }
        } else {
            self.rows_per_strip as usize
        };
        rows * self.width as usize * self.bytes_per_pixel()
    }

    /// Get number of strips
    fn num_strips(&self) -> usize {
        if self.rows_per_strip == 0 || self.rows_per_strip >= self.height {
            1
        } else {
            ((self.height as usize) + (self.rows_per_strip as usize) - 1)
                / (self.rows_per_strip as usize)
        }
    }

    /// Get number of tiles
    fn num_tiles(&self) -> (usize, usize) {
        if !self.is_tiled() {
            return (0, 0);
        }
        let tiles_x = (self.width as usize + self.tile_width as usize - 1)
            / self.tile_width as usize;
        let tiles_y = (self.height as usize + self.tile_height as usize - 1)
            / self.tile_height as usize;
        (tiles_y, tiles_x)
    }
}

/// High-performance TIFF reader
pub struct TiffReader {
    /// Memory-mapped file data
    mmap: Arc<Mmap>,
    /// File path
    path: String,
    /// Is little endian
    little_endian: bool,
    /// Is BigTIFF format
    is_bigtiff: bool,
    /// Parsed IFD entries (one per page/frame)
    ifds: Vec<IfdEntry>,
    /// Image dimensions (TCZYX)
    dimensions: Dimensions,
    /// Data type
    dtype: DType,
    /// Metadata
    metadata: Metadata,
    /// Current scene index
    current_scene: usize,
    /// Scene names and their IFD ranges
    scenes: Vec<(String, std::ops::Range<usize>)>,
}

impl TiffReader {
    /// Parse TIFF header and return (little_endian, is_bigtiff, first_ifd_offset)
    fn parse_header(data: &[u8]) -> Result<(bool, bool, u64)> {
        if data.len() < 8 {
            return Err(BioIoError::InvalidTiff("File too small".into()));
        }

        let little_endian = match &data[0..2] {
            b"II" => true,
            b"MM" => false,
            _ => return Err(BioIoError::InvalidTiff("Invalid byte order marker".into())),
        };

        let version = if little_endian {
            LittleEndian::read_u16(&data[2..4])
        } else {
            BigEndian::read_u16(&data[2..4])
        };

        let (is_bigtiff, first_ifd_offset) = match version {
            42 => {
                // Classic TIFF
                let offset = if little_endian {
                    LittleEndian::read_u32(&data[4..8]) as u64
                } else {
                    BigEndian::read_u32(&data[4..8]) as u64
                };
                (false, offset)
            }
            43 => {
                // BigTIFF
                if data.len() < 16 {
                    return Err(BioIoError::InvalidTiff("BigTIFF header too small".into()));
                }
                let offset = if little_endian {
                    LittleEndian::read_u64(&data[8..16])
                } else {
                    BigEndian::read_u64(&data[8..16])
                };
                (true, offset)
            }
            v => return Err(BioIoError::InvalidTiff(format!("Unknown TIFF version: {}", v))),
        };

        Ok((little_endian, is_bigtiff, first_ifd_offset))
    }

    /// Parse all IFD entries
    fn parse_ifds(
        data: &[u8],
        little_endian: bool,
        is_bigtiff: bool,
        first_offset: u64,
    ) -> Result<Vec<IfdEntry>> {
        let mut ifds = Vec::new();
        let mut offset = first_offset;

        while offset != 0 && offset < data.len() as u64 {
            let (ifd, next_offset) =
                Self::parse_single_ifd(data, little_endian, is_bigtiff, offset)?;
            ifds.push(ifd);
            offset = next_offset;

            // Safety limit
            if ifds.len() > 100000 {
                break;
            }
        }

        Ok(ifds)
    }

    /// Parse a single IFD
    fn parse_single_ifd(
        data: &[u8],
        little_endian: bool,
        is_bigtiff: bool,
        offset: u64,
    ) -> Result<(IfdEntry, u64)> {
        let mut cursor = Cursor::new(&data[offset as usize..]);
        let mut ifd = IfdEntry::new();

        // Read number of entries
        let num_entries = if is_bigtiff {
            if little_endian {
                cursor.read_u64::<LittleEndian>()?
            } else {
                cursor.read_u64::<BigEndian>()?
            }
        } else {
            if little_endian {
                cursor.read_u16::<LittleEndian>()? as u64
            } else {
                cursor.read_u16::<BigEndian>()? as u64
            }
        };

        // Read each tag entry
        for _ in 0..num_entries {
            Self::parse_tag(&mut cursor, data, little_endian, is_bigtiff, &mut ifd)?;
        }

        // Read next IFD offset
        let next_offset = if is_bigtiff {
            if little_endian {
                cursor.read_u64::<LittleEndian>()?
            } else {
                cursor.read_u64::<BigEndian>()?
            }
        } else {
            if little_endian {
                cursor.read_u32::<LittleEndian>()? as u64
            } else {
                cursor.read_u32::<BigEndian>()? as u64
            }
        };

        Ok((ifd, next_offset))
    }

    /// Parse a single TIFF tag
    fn parse_tag(
        cursor: &mut Cursor<&[u8]>,
        data: &[u8],
        little_endian: bool,
        is_bigtiff: bool,
        ifd: &mut IfdEntry,
    ) -> Result<()> {
        let tag = if little_endian {
            cursor.read_u16::<LittleEndian>()?
        } else {
            cursor.read_u16::<BigEndian>()?
        };

        let field_type = if little_endian {
            cursor.read_u16::<LittleEndian>()?
        } else {
            cursor.read_u16::<BigEndian>()?
        };

        let count = if is_bigtiff {
            if little_endian {
                cursor.read_u64::<LittleEndian>()?
            } else {
                cursor.read_u64::<BigEndian>()?
            }
        } else {
            if little_endian {
                cursor.read_u32::<LittleEndian>()? as u64
            } else {
                cursor.read_u32::<BigEndian>()? as u64
            }
        };

        // Read value/offset
        let value_offset_size = if is_bigtiff { 8 } else { 4 };
        let type_size = Self::get_type_size(field_type);
        let value_size = count as usize * type_size;

        let value_data = if value_size <= value_offset_size {
            // Value fits in the offset field
            let pos = cursor.position() as usize;
            cursor.seek(SeekFrom::Current(value_offset_size as i64))?;
            &data[cursor.position() as usize - value_offset_size..cursor.position() as usize]
        } else {
            // Value is at offset
            let offset = if is_bigtiff {
                if little_endian {
                    cursor.read_u64::<LittleEndian>()?
                } else {
                    cursor.read_u64::<BigEndian>()?
                }
            } else {
                if little_endian {
                    cursor.read_u32::<LittleEndian>()? as u64
                } else {
                    cursor.read_u32::<BigEndian>()? as u64
                }
            };
            if offset as usize + value_size > data.len() {
                return Ok(()); // Skip invalid tag
            }
            &data[offset as usize..offset as usize + value_size]
        };

        // Parse specific tags
        match tag {
            tags::IMAGE_WIDTH => {
                ifd.width = Self::read_value_u32(value_data, field_type, little_endian);
            }
            tags::IMAGE_HEIGHT => {
                ifd.height = Self::read_value_u32(value_data, field_type, little_endian);
            }
            tags::BITS_PER_SAMPLE => {
                ifd.bits_per_sample = Self::read_value_u16(value_data, field_type, little_endian);
            }
            tags::COMPRESSION => {
                ifd.compression = Compression::from(Self::read_value_u16(
                    value_data,
                    field_type,
                    little_endian,
                ));
            }
            tags::PHOTOMETRIC => {
                ifd.photometric = Self::read_value_u16(value_data, field_type, little_endian);
            }
            tags::SAMPLES_PER_PIXEL => {
                ifd.samples_per_pixel =
                    Self::read_value_u16(value_data, field_type, little_endian);
            }
            tags::ROWS_PER_STRIP => {
                ifd.rows_per_strip = Self::read_value_u32(value_data, field_type, little_endian);
            }
            tags::STRIP_OFFSETS => {
                ifd.strip_offsets =
                    Self::read_value_array_u64(value_data, field_type, count, little_endian);
            }
            tags::STRIP_BYTE_COUNTS => {
                ifd.strip_byte_counts =
                    Self::read_value_array_u64(value_data, field_type, count, little_endian);
            }
            tags::TILE_WIDTH => {
                ifd.tile_width = Self::read_value_u32(value_data, field_type, little_endian);
            }
            tags::TILE_HEIGHT => {
                ifd.tile_height = Self::read_value_u32(value_data, field_type, little_endian);
            }
            tags::TILE_OFFSETS => {
                ifd.tile_offsets =
                    Self::read_value_array_u64(value_data, field_type, count, little_endian);
            }
            tags::TILE_BYTE_COUNTS => {
                ifd.tile_byte_counts =
                    Self::read_value_array_u64(value_data, field_type, count, little_endian);
            }
            tags::SAMPLE_FORMAT => {
                ifd.sample_format = Self::read_value_u16(value_data, field_type, little_endian);
            }
            tags::PLANAR_CONFIG => {
                ifd.planar_config = Self::read_value_u16(value_data, field_type, little_endian);
            }
            tags::X_RESOLUTION => {
                ifd.x_resolution = Self::read_rational(value_data, little_endian);
            }
            tags::Y_RESOLUTION => {
                ifd.y_resolution = Self::read_rational(value_data, little_endian);
            }
            tags::RESOLUTION_UNIT => {
                ifd.resolution_unit = Self::read_value_u16(value_data, field_type, little_endian);
            }
            tags::IMAGE_DESCRIPTION => {
                ifd.description = Self::read_string(value_data);
            }
            tags::SOFTWARE => {
                ifd.software = Self::read_string(value_data);
            }
            _ => {}
        }

        Ok(())
    }

    fn get_type_size(field_type: u16) -> usize {
        match field_type {
            1 | 2 | 6 | 7 => 1,  // BYTE, ASCII, SBYTE, UNDEFINED
            3 | 8 => 2,          // SHORT, SSHORT
            4 | 9 | 11 => 4,     // LONG, SLONG, FLOAT
            5 | 10 | 12 => 8,    // RATIONAL, SRATIONAL, DOUBLE
            16 => 8,             // LONG8 (BigTIFF)
            17 => 8,             // SLONG8 (BigTIFF)
            18 => 8,             // IFD8 (BigTIFF)
            _ => 1,
        }
    }

    fn read_value_u16(data: &[u8], field_type: u16, little_endian: bool) -> u16 {
        if data.len() < 2 {
            return 0;
        }
        match field_type {
            1 | 6 => data[0] as u16,
            3 | 8 => {
                if little_endian {
                    LittleEndian::read_u16(data)
                } else {
                    BigEndian::read_u16(data)
                }
            }
            4 | 9 => {
                if little_endian {
                    LittleEndian::read_u32(data) as u16
                } else {
                    BigEndian::read_u32(data) as u16
                }
            }
            _ => 0,
        }
    }

    fn read_value_u32(data: &[u8], field_type: u16, little_endian: bool) -> u32 {
        match field_type {
            1 | 6 => data.get(0).copied().unwrap_or(0) as u32,
            3 | 8 if data.len() >= 2 => {
                if little_endian {
                    LittleEndian::read_u16(data) as u32
                } else {
                    BigEndian::read_u16(data) as u32
                }
            }
            4 | 9 if data.len() >= 4 => {
                if little_endian {
                    LittleEndian::read_u32(data)
                } else {
                    BigEndian::read_u32(data)
                }
            }
            16 if data.len() >= 8 => {
                if little_endian {
                    LittleEndian::read_u64(data) as u32
                } else {
                    BigEndian::read_u64(data) as u32
                }
            }
            _ => 0,
        }
    }

    fn read_value_array_u64(
        data: &[u8],
        field_type: u16,
        count: u64,
        little_endian: bool,
    ) -> Vec<u64> {
        let mut values = Vec::with_capacity(count as usize);
        let elem_size = Self::get_type_size(field_type);

        for i in 0..count as usize {
            let offset = i * elem_size;
            if offset + elem_size > data.len() {
                break;
            }
            let slice = &data[offset..];
            let value = match field_type {
                1 | 6 => slice[0] as u64,
                3 | 8 => {
                    if little_endian {
                        LittleEndian::read_u16(slice) as u64
                    } else {
                        BigEndian::read_u16(slice) as u64
                    }
                }
                4 | 9 => {
                    if little_endian {
                        LittleEndian::read_u32(slice) as u64
                    } else {
                        BigEndian::read_u32(slice) as u64
                    }
                }
                16 | 17 | 18 => {
                    if little_endian {
                        LittleEndian::read_u64(slice)
                    } else {
                        BigEndian::read_u64(slice)
                    }
                }
                _ => 0,
            };
            values.push(value);
        }
        values
    }

    fn read_rational(data: &[u8], little_endian: bool) -> Option<f64> {
        if data.len() < 8 {
            return None;
        }
        let (num, den) = if little_endian {
            (
                LittleEndian::read_u32(&data[0..4]),
                LittleEndian::read_u32(&data[4..8]),
            )
        } else {
            (
                BigEndian::read_u32(&data[0..4]),
                BigEndian::read_u32(&data[4..8]),
            )
        };
        if den == 0 {
            None
        } else {
            Some(num as f64 / den as f64)
        }
    }

    fn read_string(data: &[u8]) -> Option<String> {
        let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
        String::from_utf8(data[..end].to_vec()).ok()
    }

    /// Decompress a data block
    fn decompress(&self, data: &[u8], compression: Compression, expected_size: usize) -> Result<Vec<u8>> {
        match compression {
            Compression::None => Ok(data.to_vec()),

            Compression::Lzw => {
                use weezl::{decode::Decoder, BitOrder};
                let mut decoder = Decoder::new(BitOrder::Msb, 8);
                let mut output = Vec::with_capacity(expected_size);
                decoder
                    .into_stream(&mut output)
                    .decode_all(data)
                    .status
                    .map_err(|e| BioIoError::Decompression(format!("LZW: {:?}", e)))?;
                Ok(output)
            }

            Compression::Deflate | Compression::AdobeDeflate => {
                use flate2::read::ZlibDecoder;
                let mut decoder = ZlibDecoder::new(data);
                let mut output = Vec::with_capacity(expected_size);
                decoder
                    .read_to_end(&mut output)
                    .map_err(|e| BioIoError::Decompression(format!("Deflate: {}", e)))?;
                Ok(output)
            }

            Compression::Zstd => {
                zstd::decode_all(data).map_err(|e| BioIoError::Decompression(format!("Zstd: {}", e)))
            }

            Compression::Jpeg => {
                let mut decoder = jpeg_decoder::Decoder::new(data);
                decoder
                    .decode()
                    .map_err(|e| BioIoError::Decompression(format!("JPEG: {}", e)))
            }

            Compression::Unknown(v) => {
                Err(BioIoError::Decompression(format!("Unknown compression: {}", v)))
            }
        }
    }

    /// Read a single page with parallel strip decompression
    fn read_page_parallel(&self, page_idx: usize) -> Result<Vec<u8>> {
        let ifd = &self.ifds[page_idx];
        let height = ifd.height as usize;
        let width = ifd.width as usize;
        let bpp = ifd.bytes_per_pixel();

        if ifd.is_tiled() {
            self.read_tiles_parallel(ifd, height, width, bpp)
        } else {
            self.read_strips_parallel(ifd, height, width, bpp)
        }
    }

    /// Read stripped TIFF with parallel decompression
    fn read_strips_parallel(
        &self,
        ifd: &IfdEntry,
        height: usize,
        width: usize,
        bpp: usize,
    ) -> Result<Vec<u8>> {
        let num_strips = ifd.strip_offsets.len();
        let rows_per_strip = ifd.rows_per_strip as usize;
        let row_bytes = width * bpp;

        // Parallel strip decompression
        let strips: Vec<Result<Vec<u8>>> = (0..num_strips)
            .into_par_iter()
            .map(|strip_idx| {
                let offset = ifd.strip_offsets[strip_idx] as usize;
                let byte_count = ifd.strip_byte_counts.get(strip_idx).copied().unwrap_or(0) as usize;

                if offset + byte_count > self.mmap.len() {
                    return Err(BioIoError::InvalidTiff(format!(
                        "Strip {} extends beyond file",
                        strip_idx
                    )));
                }

                let compressed = &self.mmap[offset..offset + byte_count];
                let expected_size = ifd.strip_size(strip_idx);
                self.decompress(compressed, ifd.compression, expected_size)
            })
            .collect();

        // Combine strips into output buffer
        let mut output = vec![0u8; height * row_bytes];
        let mut row_offset = 0;

        for (strip_idx, strip_result) in strips.into_iter().enumerate() {
            let strip_data = strip_result?;
            let strip_rows = std::cmp::min(rows_per_strip, height - row_offset);
            let copy_bytes = strip_rows * row_bytes;

            if copy_bytes <= strip_data.len() {
                let dst_start = row_offset * row_bytes;
                output[dst_start..dst_start + copy_bytes]
                    .copy_from_slice(&strip_data[..copy_bytes]);
            }

            row_offset += strip_rows;
        }

        Ok(output)
    }

    /// Read tiled TIFF with parallel decompression
    fn read_tiles_parallel(
        &self,
        ifd: &IfdEntry,
        height: usize,
        width: usize,
        bpp: usize,
    ) -> Result<Vec<u8>> {
        let tile_width = ifd.tile_width as usize;
        let tile_height = ifd.tile_height as usize;
        let (tiles_y, tiles_x) = ifd.num_tiles();
        let num_tiles = ifd.tile_offsets.len();
        let row_bytes = width * bpp;

        // Parallel tile decompression
        let tiles: Vec<Result<(usize, usize, Vec<u8>)>> = (0..num_tiles)
            .into_par_iter()
            .map(|tile_idx| {
                let tile_y = tile_idx / tiles_x;
                let tile_x = tile_idx % tiles_x;

                let offset = ifd.tile_offsets[tile_idx] as usize;
                let byte_count = ifd.tile_byte_counts.get(tile_idx).copied().unwrap_or(0) as usize;

                if offset + byte_count > self.mmap.len() {
                    return Err(BioIoError::InvalidTiff(format!(
                        "Tile {} extends beyond file",
                        tile_idx
                    )));
                }

                let compressed = &self.mmap[offset..offset + byte_count];
                let expected_size = tile_width * tile_height * bpp;
                let data = self.decompress(compressed, ifd.compression, expected_size)?;

                Ok((tile_y, tile_x, data))
            })
            .collect();

        // Combine tiles into output buffer
        let mut output = vec![0u8; height * row_bytes];

        for tile_result in tiles {
            let (tile_y, tile_x, tile_data) = tile_result?;

            let y_start = tile_y * tile_height;
            let x_start = tile_x * tile_width;
            let y_end = std::cmp::min(y_start + tile_height, height);
            let x_end = std::cmp::min(x_start + tile_width, width);

            for y in y_start..y_end {
                let src_row = y - y_start;
                let src_start = src_row * tile_width * bpp;
                let src_end = src_start + (x_end - x_start) * bpp;

                if src_end <= tile_data.len() {
                    let dst_start = y * row_bytes + x_start * bpp;
                    output[dst_start..dst_start + (x_end - x_start) * bpp]
                        .copy_from_slice(&tile_data[src_start..src_end]);
                }
            }
        }

        Ok(output)
    }

    /// Build dimensions from IFDs
    fn build_dimensions(ifds: &[IfdEntry]) -> Dimensions {
        if ifds.is_empty() {
            return Dimensions::default();
        }

        let first = &ifds[0];
        let height = first.height as usize;
        let width = first.width as usize;
        let channels = first.samples_per_pixel as usize;

        // Treat each IFD as a Z slice (can be overridden by OME metadata)
        let z_slices = ifds.len();

        Dimensions::new(1, channels, z_slices, height, width)
    }

    /// Build metadata from IFDs
    fn build_metadata(ifds: &[IfdEntry]) -> Metadata {
        let mut meta = Metadata::new();

        if let Some(first) = ifds.first() {
            // Extract pixel size from resolution tags
            if let (Some(x_res), Some(y_res)) = (first.x_resolution, first.y_resolution) {
                let unit_scale = match first.resolution_unit {
                    2 => 25400.0,   // inch to micrometers
                    3 => 10000.0,   // cm to micrometers
                    _ => 1.0,       // no unit or unknown
                };

                if x_res > 0.0 {
                    meta.pixel_size_x = Some(unit_scale / x_res);
                }
                if y_res > 0.0 {
                    meta.pixel_size_y = Some(unit_scale / y_res);
                }
                meta.pixel_size_unit = Some(SizeUnit::Micrometer);
            }

            // Software
            meta.software = first.software.clone();

            // Description may contain more metadata
            if let Some(desc) = &first.description {
                meta.description = Some(desc.clone());
            }
        }

        meta
    }

    /// Determine data type from IFD
    fn determine_dtype(ifd: &IfdEntry) -> DType {
        let signed = ifd.sample_format == 2;
        let float = ifd.sample_format == 3;
        DType::from_bits_and_format(ifd.bits_per_sample, signed, float)
    }
}

impl Reader for TiffReader {
    fn open(path: &Path, options: &ReaderOptions) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let (little_endian, is_bigtiff, first_ifd) = Self::parse_header(&mmap)?;
        let ifds = Self::parse_ifds(&mmap, little_endian, is_bigtiff, first_ifd)?;

        if ifds.is_empty() {
            return Err(BioIoError::InvalidTiff("No IFD entries found".into()));
        }

        let dimensions = Self::build_dimensions(&ifds);
        let metadata = Self::build_metadata(&ifds);
        let dtype = Self::determine_dtype(&ifds[0]);

        // Default to single scene containing all IFDs
        let scenes = vec![("Scene_0".to_string(), 0..ifds.len())];

        Ok(Self {
            mmap: Arc::new(mmap),
            path: path.to_string_lossy().into_owned(),
            little_endian,
            is_bigtiff,
            ifds,
            dimensions,
            dtype,
            metadata,
            current_scene: 0,
            scenes,
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
        self.scenes.len()
    }

    fn scene_names(&self) -> Vec<String> {
        self.scenes.iter().map(|(name, _)| name.clone()).collect()
    }

    fn set_scene(&mut self, scene: usize) -> Result<()> {
        if scene >= self.scenes.len() {
            return Err(BioIoError::SceneNotFound(scene, self.scenes.len()));
        }
        self.current_scene = scene;
        Ok(())
    }

    fn current_scene(&self) -> usize {
        self.current_scene
    }

    fn read_all(&self) -> Result<ArrayD<u8>> {
        self.read_parallel(0)
    }

    fn read_region(&self, region: &Region) -> Result<ArrayD<u8>> {
        // For now, read all and slice
        // TODO: Implement direct region reading for efficiency
        let full = self.read_all()?;

        let dims = self.dimensions();
        let t_range = region.t.clone().unwrap_or(0..dims.t);
        let c_range = region.c.clone().unwrap_or(0..dims.c);
        let z_range = region.z.clone().unwrap_or(0..dims.z);
        let y_range = region.y.clone().unwrap_or(0..dims.y);
        let x_range = region.x.clone().unwrap_or(0..dims.x);

        let shape = region.shape(dims);
        let mut output = ArrayD::<u8>::zeros(IxDyn(&shape));

        // Copy the region
        for (out_t, t) in t_range.enumerate() {
            for (out_c, c) in c_range.clone().enumerate() {
                for (out_z, z) in z_range.clone().enumerate() {
                    for (out_y, y) in y_range.clone().enumerate() {
                        for (out_x, x) in x_range.clone().enumerate() {
                            output[[out_t, out_c, out_z, out_y, out_x]] =
                                full[[t, c, z, y, x]];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    fn read_parallel(&self, num_threads: usize) -> Result<ArrayD<u8>> {
        // Configure thread pool
        let pool = if num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()?
        } else {
            rayon::ThreadPoolBuilder::new().build()?
        };

        let dims = &self.dimensions;
        let bpp = self.ifds[0].bytes_per_pixel();

        // Read all pages in parallel
        let pages: Vec<Result<Vec<u8>>> = pool.install(|| {
            (0..self.ifds.len())
                .into_par_iter()
                .map(|page_idx| self.read_page_parallel(page_idx))
                .collect()
        });

        // Allocate output array
        let mut output = ArrayD::<u8>::zeros(IxDyn(&[dims.t, dims.c, dims.z, dims.y, dims.x]));

        // Copy page data into output
        for (z, page_result) in pages.into_iter().enumerate() {
            let page_data = page_result?;

            for y in 0..dims.y {
                for x in 0..dims.x {
                    let src_idx = (y * dims.x + x) * bpp;
                    // Handle multi-channel data
                    for c in 0..dims.c {
                        if src_idx + c < page_data.len() {
                            output[[0, c, z, y, x]] = page_data[src_idx + c];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    fn path(&self) -> &str {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_minimal_tiff() -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".tiff").unwrap();

        // Minimal valid TIFF with 2x2 grayscale image
        let header: Vec<u8> = vec![
            b'I', b'I',  // Little endian
            42, 0,       // TIFF magic
            8, 0, 0, 0,  // First IFD offset
        ];

        // IFD with minimal tags
        let ifd: Vec<u8> = vec![
            8, 0,  // Number of entries

            // ImageWidth (256) = 2
            0, 1, 3, 0, 1, 0, 0, 0, 2, 0, 0, 0,
            // ImageHeight (257) = 2
            1, 1, 3, 0, 1, 0, 0, 0, 2, 0, 0, 0,
            // BitsPerSample (258) = 8
            2, 1, 3, 0, 1, 0, 0, 0, 8, 0, 0, 0,
            // Compression (259) = 1 (none)
            3, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            // PhotometricInterpretation (262) = 1 (black is zero)
            6, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            // StripOffsets (273) = 122 (after IFD)
            17, 1, 4, 0, 1, 0, 0, 0, 122, 0, 0, 0,
            // SamplesPerPixel (277) = 1
            21, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            // StripByteCounts (279) = 4
            23, 1, 4, 0, 1, 0, 0, 0, 4, 0, 0, 0,

            0, 0, 0, 0,  // Next IFD = 0 (none)
        ];

        // Image data (2x2 grayscale)
        let image_data: Vec<u8> = vec![100, 150, 200, 250];

        file.write_all(&header).unwrap();
        file.write_all(&ifd).unwrap();
        file.write_all(&image_data).unwrap();
        file.flush().unwrap();

        file
    }

    #[test]
    fn test_parse_header() {
        let file = create_minimal_tiff();
        let mmap = unsafe { Mmap::map(&File::open(file.path()).unwrap()).unwrap() };

        let (le, bigtiff, offset) = TiffReader::parse_header(&mmap).unwrap();
        assert!(le);
        assert!(!bigtiff);
        assert_eq!(offset, 8);
    }
}
