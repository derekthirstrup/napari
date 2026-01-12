//! Nikon ND2 file format reader
//!
//! ND2 files use a chunk-based format with metadata stored as JSON
//! and image data in compressed chunks. This reader supports:
//! - LZ4 and Zstd compression
//! - Multi-dimensional data (TCZYX)
//! - Parallel chunk reading
//! - Metadata extraction (pixel sizes, channel names, etc.)

use crate::dimensions::Dimensions;
use crate::error::{BioIoError, Result};
use crate::format::ImageFormat;
use crate::metadata::{ChannelInfo, Metadata, SizeUnit};
use crate::readers::{DType, Reader, ReaderOptions, Region};

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

/// ND2 file magic number
const ND2_MAGIC: u32 = 0xDADADADA;

/// Chunk signature
const CHUNK_MAGIC: u32 = 0x0ADACADA;

/// Compression types used in ND2
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nd2Compression {
    None,
    Lz4,
    Zstd,
    Unknown(u32),
}

impl From<u32> for Nd2Compression {
    fn from(value: u32) -> Self {
        match value {
            0 => Nd2Compression::None,
            1 => Nd2Compression::Lz4,
            2 => Nd2Compression::Zstd,
            v => Nd2Compression::Unknown(v),
        }
    }
}

/// Chunk header information
#[derive(Debug, Clone)]
struct ChunkInfo {
    /// Offset in file
    offset: u64,
    /// Compressed size
    compressed_size: u64,
    /// Uncompressed size
    uncompressed_size: u64,
    /// Compression type
    compression: Nd2Compression,
    /// Chunk name/type
    name: String,
}

/// ND2 file attributes from metadata
#[derive(Debug, Clone, Default)]
struct Nd2Attributes {
    width: usize,
    height: usize,
    num_channels: usize,
    num_z_slices: usize,
    num_timepoints: usize,
    bits_per_component: u16,
    components_per_channel: usize,
    pixel_size_x: Option<f64>,
    pixel_size_y: Option<f64>,
    pixel_size_z: Option<f64>,
    channel_names: Vec<String>,
    channel_colors: Vec<String>,
}

/// Nikon ND2 reader
pub struct Nd2Reader {
    /// Memory-mapped file
    mmap: Arc<Mmap>,
    /// File path
    path: String,
    /// File attributes
    attributes: Nd2Attributes,
    /// Image data chunks (indexed by frame number)
    image_chunks: Vec<ChunkInfo>,
    /// Mapping from (T, C, Z) to chunk index
    frame_to_chunk: HashMap<(usize, usize, usize), usize>,
    /// Image dimensions
    dimensions: Dimensions,
    /// Data type
    dtype: DType,
    /// Metadata
    metadata: Metadata,
    /// Current scene
    current_scene: usize,
    /// Scene names
    scenes: Vec<String>,
}

impl Nd2Reader {
    /// Parse the ND2 file structure
    fn parse_file(mmap: &Mmap) -> Result<(Nd2Attributes, Vec<ChunkInfo>)> {
        if mmap.len() < 8 {
            return Err(BioIoError::InvalidNd2("File too small".into()));
        }

        // Verify magic number
        let magic = LittleEndian::read_u32(&mmap[0..4]);
        if magic != ND2_MAGIC {
            return Err(BioIoError::InvalidNd2(format!(
                "Invalid magic: expected 0x{:08X}, got 0x{:08X}",
                ND2_MAGIC, magic
            )));
        }

        // Find and parse chunk map
        let chunks = Self::find_chunks(mmap)?;

        // Extract attributes from metadata chunks
        let attributes = Self::parse_attributes(mmap, &chunks)?;

        // Filter to just image data chunks
        let image_chunks: Vec<_> = chunks
            .into_iter()
            .filter(|c| c.name.starts_with("ImageData") || c.name.contains("Image"))
            .collect();

        Ok((attributes, image_chunks))
    }

    /// Find all chunks in the file
    fn find_chunks(mmap: &Mmap) -> Result<Vec<ChunkInfo>> {
        let mut chunks = Vec::new();
        let mut cursor = Cursor::new(&mmap[..]);

        // Skip header
        cursor.seek(SeekFrom::Start(8))?;

        // Read chunk map offset from end of file
        let file_len = mmap.len() as u64;
        cursor.seek(SeekFrom::Start(file_len.saturating_sub(8)))?;
        let chunk_map_offset = cursor.read_u64::<LittleEndian>()?;

        if chunk_map_offset >= file_len {
            // Try alternative parsing - scan for chunks
            return Self::scan_for_chunks(mmap);
        }

        // Parse chunk map
        cursor.seek(SeekFrom::Start(chunk_map_offset))?;

        // Read signature
        let sig = cursor.read_u32::<LittleEndian>()?;
        if sig != CHUNK_MAGIC && sig != ND2_MAGIC {
            return Self::scan_for_chunks(mmap);
        }

        // Read chunk count
        let _unknown = cursor.read_u32::<LittleEndian>()?;
        let chunk_count = cursor.read_u64::<LittleEndian>()? as usize;

        for _ in 0..chunk_count.min(100000) {
            if cursor.position() >= file_len {
                break;
            }

            // Read chunk entry
            let offset = cursor.read_u64::<LittleEndian>()?;
            let compressed_size = cursor.read_u64::<LittleEndian>()?;
            let uncompressed_size = cursor.read_u64::<LittleEndian>()?;

            // Read name length and name
            let name_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut name_bytes = vec![0u8; name_len];
            cursor.read_exact(&mut name_bytes)?;
            let name = String::from_utf8_lossy(&name_bytes)
                .trim_end_matches('\0')
                .to_string();

            // Determine compression from chunk header
            let compression = if offset + 16 < file_len {
                let comp_byte = mmap.get(offset as usize + 12).copied().unwrap_or(0);
                Nd2Compression::from(comp_byte as u32)
            } else {
                Nd2Compression::None
            };

            chunks.push(ChunkInfo {
                offset,
                compressed_size,
                uncompressed_size,
                compression,
                name,
            });
        }

        Ok(chunks)
    }

    /// Scan file for chunks (fallback method)
    fn scan_for_chunks(mmap: &Mmap) -> Result<Vec<ChunkInfo>> {
        let mut chunks = Vec::new();
        let mut pos = 8usize; // Skip header

        while pos + 16 < mmap.len() {
            // Look for chunk signature
            let sig = LittleEndian::read_u32(&mmap[pos..pos + 4]);

            if sig == CHUNK_MAGIC {
                let compressed_size = LittleEndian::read_u64(&mmap[pos + 4..pos + 12]) as usize;
                let uncompressed_size = LittleEndian::read_u64(&mmap[pos + 12..pos + 20]) as usize;

                // Try to read name
                let name_offset = pos + 20;
                let name = if name_offset + 4 < mmap.len() {
                    let name_len = LittleEndian::read_u32(&mmap[name_offset..name_offset + 4]) as usize;
                    if name_offset + 4 + name_len < mmap.len() && name_len < 1000 {
                        String::from_utf8_lossy(&mmap[name_offset + 4..name_offset + 4 + name_len])
                            .trim_end_matches('\0')
                            .to_string()
                    } else {
                        format!("Chunk_{}", chunks.len())
                    }
                } else {
                    format!("Chunk_{}", chunks.len())
                };

                chunks.push(ChunkInfo {
                    offset: pos as u64,
                    compressed_size: compressed_size as u64,
                    uncompressed_size: uncompressed_size as u64,
                    compression: Nd2Compression::None,
                    name,
                });

                pos += 20 + compressed_size;
            } else {
                pos += 1;
            }

            if chunks.len() > 100000 {
                break;
            }
        }

        Ok(chunks)
    }

    /// Parse attributes from metadata chunks
    fn parse_attributes(mmap: &Mmap, chunks: &[ChunkInfo]) -> Result<Nd2Attributes> {
        let mut attrs = Nd2Attributes::default();

        // Find metadata chunks
        for chunk in chunks {
            if chunk.name.contains("Attributes") || chunk.name.contains("Metadata") {
                if let Ok(data) = Self::read_chunk_data(mmap, chunk) {
                    if let Ok(text) = String::from_utf8(data) {
                        Self::parse_json_metadata(&text, &mut attrs);
                    }
                }
            }
        }

        // Set defaults if not found
        if attrs.width == 0 {
            attrs.width = 512;
        }
        if attrs.height == 0 {
            attrs.height = 512;
        }
        if attrs.num_channels == 0 {
            attrs.num_channels = 1;
        }
        if attrs.num_z_slices == 0 {
            attrs.num_z_slices = 1;
        }
        if attrs.num_timepoints == 0 {
            attrs.num_timepoints = 1;
        }
        if attrs.bits_per_component == 0 {
            attrs.bits_per_component = 16;
        }

        Ok(attrs)
    }

    /// Parse JSON metadata
    fn parse_json_metadata(text: &str, attrs: &mut Nd2Attributes) {
        // Try to parse as JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
            Self::extract_from_json(&json, attrs);
        }
    }

    /// Extract attributes from JSON
    fn extract_from_json(json: &serde_json::Value, attrs: &mut Nd2Attributes) {
        // Width/Height
        if let Some(w) = json.get("widthPx").and_then(|v| v.as_u64()) {
            attrs.width = w as usize;
        }
        if let Some(h) = json.get("heightPx").and_then(|v| v.as_u64()) {
            attrs.height = h as usize;
        }

        // Alternative names
        if let Some(w) = json.get("uiWidth").and_then(|v| v.as_u64()) {
            attrs.width = w as usize;
        }
        if let Some(h) = json.get("uiHeight").and_then(|v| v.as_u64()) {
            attrs.height = h as usize;
        }

        // Bits per component
        if let Some(b) = json.get("bitsPerComponentInMemory").and_then(|v| v.as_u64()) {
            attrs.bits_per_component = b as u16;
        }
        if let Some(b) = json.get("uiBitsPerComp").and_then(|v| v.as_u64()) {
            attrs.bits_per_component = b as u16;
        }

        // Channel count
        if let Some(c) = json.get("componentCount").and_then(|v| v.as_u64()) {
            attrs.num_channels = c as usize;
        }
        if let Some(c) = json.get("uiComp").and_then(|v| v.as_u64()) {
            attrs.num_channels = c as usize;
        }

        // Sequence count (timepoints * z * positions)
        if let Some(s) = json.get("uiSequenceCount").and_then(|v| v.as_u64()) {
            // This is the total number of frames
            let total = s as usize;
            if attrs.num_z_slices == 0 {
                attrs.num_z_slices = total / attrs.num_timepoints.max(1);
            }
        }

        // Pixel sizes
        if let Some(cal) = json.get("dCalibration").and_then(|v| v.as_f64()) {
            attrs.pixel_size_x = Some(cal);
            attrs.pixel_size_y = Some(cal);
        }
        if let Some(px) = json.get("dAspect").and_then(|v| v.as_f64()) {
            if let Some(cal) = attrs.pixel_size_x {
                attrs.pixel_size_y = Some(cal * px);
            }
        }
        if let Some(z) = json.get("dZStep").and_then(|v| v.as_f64()) {
            attrs.pixel_size_z = Some(z.abs());
        }

        // Channel names
        if let Some(channels) = json.get("pPlanes").and_then(|v| v.as_array()) {
            for ch in channels {
                if let Some(name) = ch.get("sDescription").and_then(|v| v.as_str()) {
                    attrs.channel_names.push(name.to_string());
                }
            }
        }

        // Recursively search nested objects
        if let Some(obj) = json.as_object() {
            for (_, value) in obj {
                if value.is_object() || value.is_array() {
                    Self::extract_from_json(value, attrs);
                }
            }
        }
    }

    /// Read and decompress chunk data
    fn read_chunk_data(mmap: &Mmap, chunk: &ChunkInfo) -> Result<Vec<u8>> {
        let offset = chunk.offset as usize;
        let size = chunk.compressed_size as usize;

        if offset + size > mmap.len() {
            return Err(BioIoError::InvalidNd2("Chunk extends beyond file".into()));
        }

        // Skip chunk header (usually 16-32 bytes)
        let header_size = 16;
        let data_offset = offset + header_size;
        let data_size = size.saturating_sub(header_size);

        if data_offset + data_size > mmap.len() {
            return Err(BioIoError::InvalidNd2("Chunk data extends beyond file".into()));
        }

        let compressed = &mmap[data_offset..data_offset + data_size];

        match chunk.compression {
            Nd2Compression::None => Ok(compressed.to_vec()),

            Nd2Compression::Lz4 => {
                lz4_flex::decompress_size_prepended(compressed)
                    .or_else(|_| {
                        // Try without size prefix
                        let expected = chunk.uncompressed_size as usize;
                        lz4_flex::decompress(compressed, expected)
                    })
                    .map_err(|e| BioIoError::Decompression(format!("LZ4: {}", e)))
            }

            Nd2Compression::Zstd => {
                zstd::decode_all(compressed)
                    .map_err(|e| BioIoError::Decompression(format!("Zstd: {}", e)))
            }

            Nd2Compression::Unknown(v) => {
                Err(BioIoError::Decompression(format!("Unknown compression: {}", v)))
            }
        }
    }

    /// Build frame to chunk mapping
    fn build_frame_mapping(
        attrs: &Nd2Attributes,
        image_chunks: &[ChunkInfo],
    ) -> HashMap<(usize, usize, usize), usize> {
        let mut mapping = HashMap::new();

        let total_frames = attrs.num_timepoints * attrs.num_z_slices;
        let chunk_count = image_chunks.len();

        // Simple sequential mapping
        // ND2 typically stores as TZCYX order
        for (chunk_idx, _) in image_chunks.iter().enumerate() {
            if chunk_idx >= total_frames {
                break;
            }

            let t = chunk_idx / attrs.num_z_slices;
            let z = chunk_idx % attrs.num_z_slices;

            // All channels are typically in the same chunk for ND2
            for c in 0..attrs.num_channels {
                mapping.insert((t, c, z), chunk_idx);
            }
        }

        mapping
    }

    /// Read a single frame
    fn read_frame(&self, chunk_idx: usize) -> Result<Vec<u8>> {
        if chunk_idx >= self.image_chunks.len() {
            return Err(BioIoError::InvalidNd2(format!(
                "Invalid chunk index: {}",
                chunk_idx
            )));
        }

        Self::read_chunk_data(&self.mmap, &self.image_chunks[chunk_idx])
    }
}

impl Reader for Nd2Reader {
    fn open(path: &Path, _options: &ReaderOptions) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let (attributes, image_chunks) = Self::parse_file(&mmap)?;

        let dimensions = Dimensions::new(
            attributes.num_timepoints,
            attributes.num_channels,
            attributes.num_z_slices,
            attributes.height,
            attributes.width,
        );

        let dtype = match attributes.bits_per_component {
            8 => DType::U8,
            16 => DType::U16,
            32 => DType::F32,
            _ => DType::U16,
        };

        let mut metadata = Metadata::new();
        metadata.pixel_size_x = attributes.pixel_size_x;
        metadata.pixel_size_y = attributes.pixel_size_y;
        metadata.pixel_size_z = attributes.pixel_size_z;
        metadata.pixel_size_unit = Some(SizeUnit::Micrometer);
        metadata.channel_names = attributes.channel_names.clone();
        metadata.channels = attributes
            .channel_names
            .iter()
            .map(|n| ChannelInfo::new(n.clone()))
            .collect();

        let frame_to_chunk = Self::build_frame_mapping(&attributes, &image_chunks);

        Ok(Self {
            mmap: Arc::new(mmap),
            path: path.to_string_lossy().into_owned(),
            attributes,
            image_chunks,
            frame_to_chunk,
            dimensions,
            dtype,
            metadata,
            current_scene: 0,
            scenes: vec!["Scene_0".to_string()],
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
        self.scenes.clone()
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
        // Read all and slice (TODO: optimize for direct region reading)
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

    fn read_parallel(&self, num_threads: usize) -> Result<ArrayD<u8>> {
        let pool = if num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()?
        } else {
            rayon::ThreadPoolBuilder::new().build()?
        };

        let dims = &self.dimensions;
        let bytes_per_pixel = self.dtype.size_bytes();
        let frame_size = dims.y * dims.x * dims.c * bytes_per_pixel;

        // Collect unique chunk indices
        let chunk_indices: Vec<usize> = (0..self.image_chunks.len()).collect();

        // Read all chunks in parallel
        let chunk_data: Vec<Result<(usize, Vec<u8>)>> = pool.install(|| {
            chunk_indices
                .par_iter()
                .map(|&idx| {
                    let data = self.read_frame(idx)?;
                    Ok((idx, data))
                })
                .collect()
        });

        // Allocate output array
        let mut output = ArrayD::<u8>::zeros(IxDyn(&[dims.t, dims.c, dims.z, dims.y, dims.x]));

        // Copy data into output array
        for result in chunk_data {
            let (chunk_idx, data) = result?;

            // Find which frames this chunk corresponds to
            for (&(t, c, z), &cidx) in &self.frame_to_chunk {
                if cidx == chunk_idx {
                    // Extract this channel's data from the chunk
                    let channel_offset = c * dims.y * dims.x * bytes_per_pixel;

                    for y in 0..dims.y {
                        for x in 0..dims.x {
                            let src_idx = channel_offset + (y * dims.x + x) * bytes_per_pixel;

                            if src_idx < data.len() {
                                // For u8 data
                                if bytes_per_pixel == 1 {
                                    output[[t, c, z, y, x]] = data[src_idx];
                                } else if bytes_per_pixel == 2 && src_idx + 1 < data.len() {
                                    // For u16 data, store high byte
                                    output[[t, c, z, y, x]] = data[src_idx + 1];
                                }
                            }
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

    #[test]
    fn test_nd2_compression_from() {
        assert_eq!(Nd2Compression::from(0), Nd2Compression::None);
        assert_eq!(Nd2Compression::from(1), Nd2Compression::Lz4);
        assert_eq!(Nd2Compression::from(2), Nd2Compression::Zstd);
        assert_eq!(Nd2Compression::from(99), Nd2Compression::Unknown(99));
    }

    #[test]
    fn test_json_metadata_parsing() {
        let mut attrs = Nd2Attributes::default();
        let json = r#"{
            "widthPx": 1024,
            "heightPx": 512,
            "bitsPerComponentInMemory": 16,
            "componentCount": 2,
            "dCalibration": 0.1,
            "dZStep": 0.5
        }"#;

        Nd2Reader::parse_json_metadata(json, &mut attrs);

        assert_eq!(attrs.width, 1024);
        assert_eq!(attrs.height, 512);
        assert_eq!(attrs.bits_per_component, 16);
        assert_eq!(attrs.num_channels, 2);
        assert_eq!(attrs.pixel_size_x, Some(0.1));
        assert_eq!(attrs.pixel_size_z, Some(0.5));
    }
}
