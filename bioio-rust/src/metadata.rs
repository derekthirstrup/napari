//! Image metadata handling
//!
//! Provides a unified metadata model for microscopy images,
//! with support for format-specific metadata parsing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Physical size with unit
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalSize {
    pub value: f64,
    pub unit: SizeUnit,
}

impl PhysicalSize {
    pub fn new(value: f64, unit: SizeUnit) -> Self {
        Self { value, unit }
    }

    /// Convert to micrometers
    pub fn to_micrometers(&self) -> f64 {
        match self.unit {
            SizeUnit::Nanometer => self.value / 1000.0,
            SizeUnit::Micrometer => self.value,
            SizeUnit::Millimeter => self.value * 1000.0,
            SizeUnit::Centimeter => self.value * 10000.0,
            SizeUnit::Meter => self.value * 1_000_000.0,
            SizeUnit::Inch => self.value * 25400.0,
            SizeUnit::Pixel => self.value, // No conversion possible
        }
    }
}

/// Size units
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SizeUnit {
    Nanometer,
    Micrometer,
    Millimeter,
    Centimeter,
    Meter,
    Inch,
    Pixel,
}

impl SizeUnit {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "nm" | "nanometer" | "nanometers" => SizeUnit::Nanometer,
            "um" | "µm" | "micrometer" | "micrometers" | "micron" | "microns" => {
                SizeUnit::Micrometer
            }
            "mm" | "millimeter" | "millimeters" => SizeUnit::Millimeter,
            "cm" | "centimeter" | "centimeters" => SizeUnit::Centimeter,
            "m" | "meter" | "meters" => SizeUnit::Meter,
            "in" | "inch" | "inches" => SizeUnit::Inch,
            _ => SizeUnit::Pixel,
        }
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            SizeUnit::Nanometer => "nm",
            SizeUnit::Micrometer => "µm",
            SizeUnit::Millimeter => "mm",
            SizeUnit::Centimeter => "cm",
            SizeUnit::Meter => "m",
            SizeUnit::Inch => "in",
            SizeUnit::Pixel => "px",
        }
    }
}

/// Channel information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub name: String,
    pub color: Option<String>,
    pub emission_wavelength: Option<f64>,
    pub excitation_wavelength: Option<f64>,
    pub exposure_time: Option<f64>,
    pub contrast_limits: Option<(f64, f64)>,
}

impl ChannelInfo {
    pub fn new(name: String) -> Self {
        Self {
            name,
            color: None,
            emission_wavelength: None,
            excitation_wavelength: None,
            exposure_time: None,
            contrast_limits: None,
        }
    }
}

impl Default for ChannelInfo {
    fn default() -> Self {
        Self::new("Channel".to_string())
    }
}

/// Objective lens information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveInfo {
    pub name: Option<String>,
    pub magnification: Option<f64>,
    pub numerical_aperture: Option<f64>,
    pub immersion: Option<String>,
}

impl Default for ObjectiveInfo {
    fn default() -> Self {
        Self {
            name: None,
            magnification: None,
            numerical_aperture: None,
            immersion: None,
        }
    }
}

/// Image metadata container
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    // Physical dimensions
    pub pixel_size_x: Option<f64>,
    pub pixel_size_y: Option<f64>,
    pub pixel_size_z: Option<f64>,
    pub pixel_size_unit: Option<SizeUnit>,

    // Time information
    pub time_interval: Option<f64>,
    pub time_unit: Option<String>,

    // Channel information
    pub channel_names: Vec<String>,
    pub channels: Vec<ChannelInfo>,

    // Acquisition info
    pub objective: Option<ObjectiveInfo>,
    pub acquisition_date: Option<String>,
    pub instrument: Option<String>,
    pub software: Option<String>,

    // Image description
    pub description: Option<String>,
    pub image_name: Option<String>,

    // Format-specific raw metadata
    pub raw: HashMap<String, serde_json::Value>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set pixel sizes (in micrometers by default)
    pub fn with_pixel_size(mut self, x: f64, y: f64, z: Option<f64>) -> Self {
        self.pixel_size_x = Some(x);
        self.pixel_size_y = Some(y);
        self.pixel_size_z = z;
        self.pixel_size_unit = Some(SizeUnit::Micrometer);
        self
    }

    /// Set channel names
    pub fn with_channels(mut self, names: Vec<String>) -> Self {
        self.channel_names = names.clone();
        self.channels = names.into_iter().map(ChannelInfo::new).collect();
        self
    }

    /// Add raw metadata key-value pair
    pub fn add_raw(&mut self, key: &str, value: serde_json::Value) {
        self.raw.insert(key.to_string(), value);
    }

    /// Get pixel size as PhysicalSize
    pub fn get_physical_pixel_size_x(&self) -> Option<PhysicalSize> {
        self.pixel_size_x.map(|v| PhysicalSize {
            value: v,
            unit: self.pixel_size_unit.unwrap_or(SizeUnit::Micrometer),
        })
    }

    /// Get pixel size as PhysicalSize
    pub fn get_physical_pixel_size_y(&self) -> Option<PhysicalSize> {
        self.pixel_size_y.map(|v| PhysicalSize {
            value: v,
            unit: self.pixel_size_unit.unwrap_or(SizeUnit::Micrometer),
        })
    }

    /// Get pixel size as PhysicalSize
    pub fn get_physical_pixel_size_z(&self) -> Option<PhysicalSize> {
        self.pixel_size_z.map(|v| PhysicalSize {
            value: v,
            unit: self.pixel_size_unit.unwrap_or(SizeUnit::Micrometer),
        })
    }

    /// Get voxel size as (x, y, z) tuple in micrometers
    pub fn voxel_size_um(&self) -> Option<(f64, f64, f64)> {
        match (self.pixel_size_x, self.pixel_size_y, self.pixel_size_z) {
            (Some(x), Some(y), Some(z)) => {
                let unit = self.pixel_size_unit.unwrap_or(SizeUnit::Micrometer);
                let ps_x = PhysicalSize::new(x, unit);
                let ps_y = PhysicalSize::new(y, unit);
                let ps_z = PhysicalSize::new(z, unit);
                Some((
                    ps_x.to_micrometers(),
                    ps_y.to_micrometers(),
                    ps_z.to_micrometers(),
                ))
            }
            _ => None,
        }
    }

    /// Merge with another metadata object (other takes precedence)
    pub fn merge(&mut self, other: &Metadata) {
        if other.pixel_size_x.is_some() {
            self.pixel_size_x = other.pixel_size_x;
        }
        if other.pixel_size_y.is_some() {
            self.pixel_size_y = other.pixel_size_y;
        }
        if other.pixel_size_z.is_some() {
            self.pixel_size_z = other.pixel_size_z;
        }
        if other.pixel_size_unit.is_some() {
            self.pixel_size_unit = other.pixel_size_unit;
        }
        if !other.channel_names.is_empty() {
            self.channel_names = other.channel_names.clone();
        }
        if !other.channels.is_empty() {
            self.channels = other.channels.clone();
        }
        if other.objective.is_some() {
            self.objective = other.objective.clone();
        }
        if other.description.is_some() {
            self.description = other.description.clone();
        }

        // Merge raw metadata
        for (k, v) in &other.raw {
            self.raw.insert(k.clone(), v.clone());
        }
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Create from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl std::fmt::Display for Metadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Metadata {{")?;
        if let Some(x) = self.pixel_size_x {
            let unit = self.pixel_size_unit.unwrap_or(SizeUnit::Micrometer);
            writeln!(f, "  pixel_size_x: {} {}", x, unit.symbol())?;
        }
        if let Some(y) = self.pixel_size_y {
            let unit = self.pixel_size_unit.unwrap_or(SizeUnit::Micrometer);
            writeln!(f, "  pixel_size_y: {} {}", y, unit.symbol())?;
        }
        if let Some(z) = self.pixel_size_z {
            let unit = self.pixel_size_unit.unwrap_or(SizeUnit::Micrometer);
            writeln!(f, "  pixel_size_z: {} {}", z, unit.symbol())?;
        }
        if !self.channel_names.is_empty() {
            writeln!(f, "  channels: {:?}", self.channel_names)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_size_conversion() {
        let size_nm = PhysicalSize::new(100.0, SizeUnit::Nanometer);
        assert!((size_nm.to_micrometers() - 0.1).abs() < 1e-10);

        let size_mm = PhysicalSize::new(1.0, SizeUnit::Millimeter);
        assert!((size_mm.to_micrometers() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_metadata_builder() {
        let meta = Metadata::new()
            .with_pixel_size(0.1, 0.1, Some(0.5))
            .with_channels(vec!["DAPI".to_string(), "GFP".to_string()]);

        assert_eq!(meta.pixel_size_x, Some(0.1));
        assert_eq!(meta.channel_names, vec!["DAPI", "GFP"]);
    }

    #[test]
    fn test_metadata_json() {
        let meta = Metadata::new().with_pixel_size(0.1, 0.1, None);

        let json = meta.to_json().unwrap();
        let parsed = Metadata::from_json(&json).unwrap();

        assert_eq!(parsed.pixel_size_x, meta.pixel_size_x);
    }
}
