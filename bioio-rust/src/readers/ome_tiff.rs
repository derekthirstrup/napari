//! OME-TIFF reader with XML metadata parsing
//!
//! OME-TIFF extends standard TIFF with structured XML metadata
//! describing the biological/microscopy context of the image data.
//! This reader:
//! - Parses OME-XML from the ImageDescription tag
//! - Extracts physical pixel sizes, channel info, etc.
//! - Correctly interprets multi-dimensional data ordering
//! - Falls back to standard TIFF reading for image data

use crate::dimensions::Dimensions;
use crate::error::{BioIoError, Result};
use crate::format::ImageFormat;
use crate::metadata::{ChannelInfo, Metadata, ObjectiveInfo, SizeUnit};
use crate::readers::tiff::TiffReader;
use crate::readers::{DType, Reader, ReaderOptions, Region};

use ndarray::ArrayD;
use roxmltree::{Document, Node};
use std::path::Path;

/// OME-TIFF reader
pub struct OmeTiffReader {
    /// Underlying TIFF reader for image data
    tiff_reader: TiffReader,
    /// OME metadata parsed from XML
    ome_metadata: OmeMetadata,
    /// Overridden dimensions from OME-XML
    dimensions: Dimensions,
    /// Enhanced metadata
    metadata: Metadata,
}

/// Parsed OME-XML metadata
#[derive(Debug, Clone, Default)]
struct OmeMetadata {
    /// Image name
    name: Option<String>,
    /// Acquisition date
    acquisition_date: Option<String>,
    /// Physical size X (in microns)
    physical_size_x: Option<f64>,
    /// Physical size Y (in microns)
    physical_size_y: Option<f64>,
    /// Physical size Z (in microns)
    physical_size_z: Option<f64>,
    /// Physical size unit
    physical_size_unit: SizeUnit,
    /// Time increment
    time_increment: Option<f64>,
    /// Dimension order (e.g., "XYZCT")
    dimension_order: String,
    /// Size in X
    size_x: usize,
    /// Size in Y
    size_y: usize,
    /// Size in Z
    size_z: usize,
    /// Size in C (channels)
    size_c: usize,
    /// Size in T (timepoints)
    size_t: usize,
    /// Channel information
    channels: Vec<OmeChannel>,
    /// Objective information
    objective: Option<ObjectiveInfo>,
    /// Multiple images/scenes
    images: Vec<OmeImage>,
}

#[derive(Debug, Clone, Default)]
struct OmeChannel {
    id: String,
    name: Option<String>,
    color: Option<String>,
    emission_wavelength: Option<f64>,
    excitation_wavelength: Option<f64>,
    samples_per_pixel: usize,
}

#[derive(Debug, Clone, Default)]
struct OmeImage {
    id: String,
    name: Option<String>,
    pixels: OmePixels,
}

#[derive(Debug, Clone, Default)]
struct OmePixels {
    dimension_order: String,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    size_c: usize,
    size_t: usize,
    physical_size_x: Option<f64>,
    physical_size_y: Option<f64>,
    physical_size_z: Option<f64>,
    channels: Vec<OmeChannel>,
    tiff_data: Vec<TiffDataEntry>,
}

#[derive(Debug, Clone, Default)]
struct TiffDataEntry {
    ifd: usize,
    first_c: usize,
    first_t: usize,
    first_z: usize,
    plane_count: usize,
}

impl OmeTiffReader {
    /// Parse OME-XML from the ImageDescription tag
    fn parse_ome_xml(xml: &str) -> Result<OmeMetadata> {
        let doc = Document::parse(xml).map_err(|e| BioIoError::XmlParse(e.to_string()))?;

        let root = doc.root_element();
        let mut metadata = OmeMetadata::default();

        // Find OME element (might be root or nested)
        let ome_elem = if root.tag_name().name() == "OME" {
            Some(root)
        } else {
            root.descendants().find(|n| n.tag_name().name() == "OME")
        };

        if let Some(ome) = ome_elem {
            // Parse all Image elements
            for image_node in ome.children().filter(|n| n.tag_name().name() == "Image") {
                let image = Self::parse_image_element(&image_node)?;
                metadata.images.push(image);
            }

            // Use first image as default
            if let Some(first_image) = metadata.images.first() {
                metadata.name = first_image.name.clone();
                metadata.dimension_order = first_image.pixels.dimension_order.clone();
                metadata.size_x = first_image.pixels.size_x;
                metadata.size_y = first_image.pixels.size_y;
                metadata.size_z = first_image.pixels.size_z;
                metadata.size_c = first_image.pixels.size_c;
                metadata.size_t = first_image.pixels.size_t;
                metadata.physical_size_x = first_image.pixels.physical_size_x;
                metadata.physical_size_y = first_image.pixels.physical_size_y;
                metadata.physical_size_z = first_image.pixels.physical_size_z;
                metadata.channels = first_image.pixels.channels.clone();
            }

            // Parse Instrument element
            if let Some(instrument) = ome.children().find(|n| n.tag_name().name() == "Instrument") {
                metadata.objective = Self::parse_objective(&instrument);
            }
        }

        // Set defaults
        if metadata.size_x == 0 {
            metadata.size_x = 1;
        }
        if metadata.size_y == 0 {
            metadata.size_y = 1;
        }
        if metadata.size_z == 0 {
            metadata.size_z = 1;
        }
        if metadata.size_c == 0 {
            metadata.size_c = 1;
        }
        if metadata.size_t == 0 {
            metadata.size_t = 1;
        }
        if metadata.dimension_order.is_empty() {
            metadata.dimension_order = "XYZCT".to_string();
        }

        Ok(metadata)
    }

    /// Parse an Image element
    fn parse_image_element(node: &Node) -> Result<OmeImage> {
        let mut image = OmeImage::default();

        image.id = node.attribute("ID").unwrap_or("").to_string();
        image.name = node.attribute("Name").map(String::from);

        // Find Pixels element
        if let Some(pixels_node) = node.children().find(|n| n.tag_name().name() == "Pixels") {
            image.pixels = Self::parse_pixels_element(&pixels_node)?;
        }

        Ok(image)
    }

    /// Parse a Pixels element
    fn parse_pixels_element(node: &Node) -> Result<OmePixels> {
        let mut pixels = OmePixels::default();

        // Parse attributes
        pixels.dimension_order = node.attribute("DimensionOrder").unwrap_or("XYZCT").to_string();

        pixels.size_x = node
            .attribute("SizeX")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        pixels.size_y = node
            .attribute("SizeY")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        pixels.size_z = node
            .attribute("SizeZ")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        pixels.size_c = node
            .attribute("SizeC")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        pixels.size_t = node
            .attribute("SizeT")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        // Physical sizes
        pixels.physical_size_x = node.attribute("PhysicalSizeX").and_then(|s| s.parse().ok());
        pixels.physical_size_y = node.attribute("PhysicalSizeY").and_then(|s| s.parse().ok());
        pixels.physical_size_z = node.attribute("PhysicalSizeZ").and_then(|s| s.parse().ok());

        // Parse Channel elements
        for channel_node in node.children().filter(|n| n.tag_name().name() == "Channel") {
            let channel = Self::parse_channel_element(&channel_node);
            pixels.channels.push(channel);
        }

        // Parse TiffData elements
        for tiff_node in node.children().filter(|n| n.tag_name().name() == "TiffData") {
            let tiff_data = Self::parse_tiff_data(&tiff_node);
            pixels.tiff_data.push(tiff_data);
        }

        Ok(pixels)
    }

    /// Parse a Channel element
    fn parse_channel_element(node: &Node) -> OmeChannel {
        OmeChannel {
            id: node.attribute("ID").unwrap_or("").to_string(),
            name: node.attribute("Name").map(String::from),
            color: node.attribute("Color").map(String::from),
            emission_wavelength: node
                .attribute("EmissionWavelength")
                .and_then(|s| s.parse().ok()),
            excitation_wavelength: node
                .attribute("ExcitationWavelength")
                .and_then(|s| s.parse().ok()),
            samples_per_pixel: node
                .attribute("SamplesPerPixel")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
        }
    }

    /// Parse a TiffData element
    fn parse_tiff_data(node: &Node) -> TiffDataEntry {
        TiffDataEntry {
            ifd: node
                .attribute("IFD")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            first_c: node
                .attribute("FirstC")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            first_t: node
                .attribute("FirstT")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            first_z: node
                .attribute("FirstZ")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            plane_count: node
                .attribute("PlaneCount")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
        }
    }

    /// Parse Objective element from Instrument
    fn parse_objective(instrument: &Node) -> Option<ObjectiveInfo> {
        let objective_node = instrument
            .children()
            .find(|n| n.tag_name().name() == "Objective")?;

        Some(ObjectiveInfo {
            name: objective_node.attribute("Model").map(String::from),
            magnification: objective_node
                .attribute("NominalMagnification")
                .and_then(|s| s.parse().ok()),
            numerical_aperture: objective_node
                .attribute("LensNA")
                .and_then(|s| s.parse().ok()),
            immersion: objective_node.attribute("Immersion").map(String::from),
        })
    }

    /// Convert OME metadata to standard Metadata
    fn build_metadata(ome: &OmeMetadata) -> Metadata {
        let mut meta = Metadata::new();

        meta.image_name = ome.name.clone();
        meta.acquisition_date = ome.acquisition_date.clone();
        meta.pixel_size_x = ome.physical_size_x;
        meta.pixel_size_y = ome.physical_size_y;
        meta.pixel_size_z = ome.physical_size_z;
        meta.pixel_size_unit = Some(ome.physical_size_unit);
        meta.time_interval = ome.time_increment;
        meta.objective = ome.objective.clone();

        // Build channel info
        meta.channel_names = ome
            .channels
            .iter()
            .map(|c| c.name.clone().unwrap_or_else(|| c.id.clone()))
            .collect();

        meta.channels = ome
            .channels
            .iter()
            .map(|c| ChannelInfo {
                name: c.name.clone().unwrap_or_else(|| c.id.clone()),
                color: c.color.clone(),
                emission_wavelength: c.emission_wavelength,
                excitation_wavelength: c.excitation_wavelength,
                exposure_time: None,
                contrast_limits: None,
            })
            .collect();

        meta
    }
}

impl Reader for OmeTiffReader {
    fn open(path: &Path, options: &ReaderOptions) -> Result<Self> {
        // Open as standard TIFF first
        let tiff_reader = TiffReader::open(path, options)?;

        // Try to parse OME-XML from metadata
        let ome_metadata = if let Some(desc) = &tiff_reader.metadata().description {
            if desc.contains("<OME") || desc.contains("ome.xsd") {
                Self::parse_ome_xml(desc).unwrap_or_default()
            } else {
                OmeMetadata::default()
            }
        } else {
            OmeMetadata::default()
        };

        // Build dimensions from OME metadata if available
        let dimensions = if ome_metadata.size_x > 0 {
            Dimensions::new(
                ome_metadata.size_t,
                ome_metadata.size_c,
                ome_metadata.size_z,
                ome_metadata.size_y,
                ome_metadata.size_x,
            )
        } else {
            tiff_reader.dimensions().clone()
        };

        let metadata = Self::build_metadata(&ome_metadata);

        Ok(Self {
            tiff_reader,
            ome_metadata,
            dimensions,
            metadata,
        })
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::OmeTiff
    }

    fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    fn dtype(&self) -> DType {
        self.tiff_reader.dtype()
    }

    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn num_scenes(&self) -> usize {
        self.ome_metadata.images.len().max(1)
    }

    fn scene_names(&self) -> Vec<String> {
        if self.ome_metadata.images.is_empty() {
            vec!["Scene_0".to_string()]
        } else {
            self.ome_metadata
                .images
                .iter()
                .enumerate()
                .map(|(i, img)| img.name.clone().unwrap_or_else(|| format!("Scene_{}", i)))
                .collect()
        }
    }

    fn set_scene(&mut self, scene: usize) -> Result<()> {
        if scene >= self.num_scenes() {
            return Err(BioIoError::SceneNotFound(scene, self.num_scenes()));
        }

        // Update dimensions for the selected scene
        if let Some(image) = self.ome_metadata.images.get(scene) {
            self.dimensions = Dimensions::new(
                image.pixels.size_t,
                image.pixels.size_c,
                image.pixels.size_z,
                image.pixels.size_y,
                image.pixels.size_x,
            );
        }

        self.tiff_reader.set_scene(scene)
    }

    fn current_scene(&self) -> usize {
        self.tiff_reader.current_scene()
    }

    fn read_all(&self) -> Result<ArrayD<u8>> {
        // Use TIFF reader for actual data, reshape according to OME dimensions
        let data = self.tiff_reader.read_all()?;

        // If dimensions match, return as-is
        let tiff_dims = self.tiff_reader.dimensions();
        if tiff_dims.total_pixels() == self.dimensions.total_pixels() {
            // Reshape if needed
            let shape = self.dimensions.shape_vec();
            if data.shape() != shape.as_slice() {
                return Ok(data
                    .into_shape_with_order(ndarray::IxDyn(&shape))
                    .map_err(|e| BioIoError::Other(e.to_string()))?);
            }
        }

        Ok(data)
    }

    fn read_region(&self, region: &Region) -> Result<ArrayD<u8>> {
        self.tiff_reader.read_region(region)
    }

    fn read_parallel(&self, num_threads: usize) -> Result<ArrayD<u8>> {
        let data = self.tiff_reader.read_parallel(num_threads)?;

        // Reshape according to OME dimensions if needed
        let shape = self.dimensions.shape_vec();
        if data.shape() != shape.as_slice() && data.len() == self.dimensions.total_pixels() {
            return Ok(data
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .map_err(|e| BioIoError::Other(e.to_string()))?);
        }

        Ok(data)
    }

    fn path(&self) -> &str {
        self.tiff_reader.path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_OME_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
    <Image ID="Image:0" Name="Test Image">
        <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
                SizeX="512" SizeY="512" SizeZ="10" SizeC="2" SizeT="5"
                PhysicalSizeX="0.1" PhysicalSizeY="0.1" PhysicalSizeZ="0.5">
            <Channel ID="Channel:0:0" Name="DAPI" EmissionWavelength="461"/>
            <Channel ID="Channel:0:1" Name="GFP" EmissionWavelength="509"/>
            <TiffData IFD="0" FirstC="0" FirstT="0" FirstZ="0"/>
        </Pixels>
    </Image>
</OME>"#;

    #[test]
    fn test_parse_ome_xml() {
        let metadata = OmeTiffReader::parse_ome_xml(SAMPLE_OME_XML).unwrap();

        assert_eq!(metadata.size_x, 512);
        assert_eq!(metadata.size_y, 512);
        assert_eq!(metadata.size_z, 10);
        assert_eq!(metadata.size_c, 2);
        assert_eq!(metadata.size_t, 5);
        assert_eq!(metadata.physical_size_x, Some(0.1));
        assert_eq!(metadata.physical_size_z, Some(0.5));
        assert_eq!(metadata.channels.len(), 2);
        assert_eq!(metadata.channels[0].name, Some("DAPI".to_string()));
        assert_eq!(metadata.channels[1].emission_wavelength, Some(509.0));
    }

    #[test]
    fn test_parse_channel_element() {
        let xml = r#"<Channel ID="Channel:0" Name="DAPI" Color="-16776961" EmissionWavelength="461.0" SamplesPerPixel="1"/>"#;
        let doc = Document::parse(xml).unwrap();
        let node = doc.root_element();

        let channel = OmeTiffReader::parse_channel_element(&node);

        assert_eq!(channel.id, "Channel:0");
        assert_eq!(channel.name, Some("DAPI".to_string()));
        assert_eq!(channel.emission_wavelength, Some(461.0));
        assert_eq!(channel.samples_per_pixel, 1);
    }
}
