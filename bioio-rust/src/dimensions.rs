//! Image dimension handling
//!
//! Provides unified dimension representation for microscopy images
//! using the standard TCZYX (Time, Channel, Z, Y, X) ordering.

use serde::{Deserialize, Serialize};

/// Standard dimension ordering for microscopy images
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DimensionOrder {
    /// Time, Channel, Z, Y, X (standard bioio order)
    TCZYX,
    /// Time, Z, Channel, Y, X
    TZCYX,
    /// Channel, Time, Z, Y, X
    CTZYX,
    /// Z, Y, X (3D volume)
    ZYX,
    /// Channel, Y, X (multichannel 2D)
    CYX,
    /// Y, X (2D image)
    YX,
    /// Custom ordering
    Custom(char, char, char, char, char),
}

impl DimensionOrder {
    /// Parse dimension order from string (e.g., "TCZYX", "ZYX")
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TCZYX" => Some(DimensionOrder::TCZYX),
            "TZCYX" => Some(DimensionOrder::TZCYX),
            "CTZYX" => Some(DimensionOrder::CTZYX),
            "ZYX" => Some(DimensionOrder::ZYX),
            "CYX" => Some(DimensionOrder::CYX),
            "YX" => Some(DimensionOrder::YX),
            s if s.len() == 5 => {
                let chars: Vec<char> = s.chars().collect();
                Some(DimensionOrder::Custom(
                    chars[0], chars[1], chars[2], chars[3], chars[4],
                ))
            }
            _ => None,
        }
    }

    /// Get dimension names as string
    pub fn as_str(&self) -> &'static str {
        match self {
            DimensionOrder::TCZYX => "TCZYX",
            DimensionOrder::TZCYX => "TZCYX",
            DimensionOrder::CTZYX => "CTZYX",
            DimensionOrder::ZYX => "ZYX",
            DimensionOrder::CYX => "CYX",
            DimensionOrder::YX => "YX",
            DimensionOrder::Custom(_, _, _, _, _) => "Custom",
        }
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        match self {
            DimensionOrder::TCZYX
            | DimensionOrder::TZCYX
            | DimensionOrder::CTZYX
            | DimensionOrder::Custom(_, _, _, _, _) => 5,
            DimensionOrder::ZYX => 3,
            DimensionOrder::CYX => 3,
            DimensionOrder::YX => 2,
        }
    }
}

impl Default for DimensionOrder {
    fn default() -> Self {
        DimensionOrder::TCZYX
    }
}

/// Image dimensions with optional labels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dimensions {
    /// Time points
    pub t: usize,
    /// Channels
    pub c: usize,
    /// Z slices
    pub z: usize,
    /// Height (Y)
    pub y: usize,
    /// Width (X)
    pub x: usize,
    /// Dimension ordering
    pub order: DimensionOrder,
}

impl Dimensions {
    /// Create new dimensions with TCZYX ordering
    pub fn new(t: usize, c: usize, z: usize, y: usize, x: usize) -> Self {
        Self {
            t,
            c,
            z,
            y,
            x,
            order: DimensionOrder::TCZYX,
        }
    }

    /// Create dimensions for a 2D image
    pub fn new_2d(height: usize, width: usize) -> Self {
        Self {
            t: 1,
            c: 1,
            z: 1,
            y: height,
            x: width,
            order: DimensionOrder::YX,
        }
    }

    /// Create dimensions for a 3D volume
    pub fn new_3d(z: usize, y: usize, x: usize) -> Self {
        Self {
            t: 1,
            c: 1,
            z,
            y,
            x,
            order: DimensionOrder::ZYX,
        }
    }

    /// Create dimensions with custom ordering
    pub fn with_order(mut self, order: DimensionOrder) -> Self {
        self.order = order;
        self
    }

    /// Get shape as slice in TCZYX order
    pub fn shape(&self) -> [usize; 5] {
        [self.t, self.c, self.z, self.y, self.x]
    }

    /// Get shape as vector (useful for ndarray)
    pub fn shape_vec(&self) -> Vec<usize> {
        vec![self.t, self.c, self.z, self.y, self.x]
    }

    /// Get effective shape (excluding dimensions of size 1)
    pub fn effective_shape(&self) -> Vec<usize> {
        let mut shape = Vec::new();
        if self.t > 1 {
            shape.push(self.t);
        }
        if self.c > 1 {
            shape.push(self.c);
        }
        if self.z > 1 {
            shape.push(self.z);
        }
        shape.push(self.y);
        shape.push(self.x);
        shape
    }

    /// Total number of pixels
    pub fn total_pixels(&self) -> usize {
        self.t * self.c * self.z * self.y * self.x
    }

    /// Total number of bytes for a given data type size
    pub fn total_bytes(&self, bytes_per_pixel: usize) -> usize {
        self.total_pixels() * bytes_per_pixel
    }

    /// Number of 2D frames (T * C * Z)
    pub fn num_frames(&self) -> usize {
        self.t * self.c * self.z
    }

    /// Size of a single 2D frame in pixels
    pub fn frame_size(&self) -> usize {
        self.y * self.x
    }

    /// Check if this is a 2D image
    pub fn is_2d(&self) -> bool {
        self.t == 1 && self.c == 1 && self.z == 1
    }

    /// Check if this is a 3D volume (single timepoint, single channel)
    pub fn is_3d(&self) -> bool {
        self.t == 1 && self.c == 1 && self.z > 1
    }

    /// Check if this has multiple timepoints
    pub fn is_timelapse(&self) -> bool {
        self.t > 1
    }

    /// Check if this has multiple channels
    pub fn is_multichannel(&self) -> bool {
        self.c > 1
    }

    /// Get dimension value by name
    pub fn get(&self, dim: char) -> Option<usize> {
        match dim.to_ascii_uppercase() {
            'T' => Some(self.t),
            'C' => Some(self.c),
            'Z' => Some(self.z),
            'Y' => Some(self.y),
            'X' => Some(self.x),
            _ => None,
        }
    }

    /// Calculate the linear index for a given TCZYX coordinate
    pub fn linear_index(&self, t: usize, c: usize, z: usize, y: usize, x: usize) -> usize {
        ((((t * self.c + c) * self.z + z) * self.y + y) * self.x) + x
    }

    /// Convert linear index to TCZYX coordinates
    pub fn coords_from_linear(&self, index: usize) -> (usize, usize, usize, usize, usize) {
        let x = index % self.x;
        let remaining = index / self.x;
        let y = remaining % self.y;
        let remaining = remaining / self.y;
        let z = remaining % self.z;
        let remaining = remaining / self.z;
        let c = remaining % self.c;
        let t = remaining / self.c;
        (t, c, z, y, x)
    }
}

impl Default for Dimensions {
    fn default() -> Self {
        Self::new(1, 1, 1, 1, 1)
    }
}

impl std::fmt::Display for Dimensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Dimensions(T={}, C={}, Z={}, Y={}, X={})",
            self.t, self.c, self.z, self.y, self.x
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions_shape() {
        let dims = Dimensions::new(2, 3, 4, 100, 200);
        assert_eq!(dims.shape(), [2, 3, 4, 100, 200]);
        assert_eq!(dims.total_pixels(), 2 * 3 * 4 * 100 * 200);
    }

    #[test]
    fn test_dimensions_2d() {
        let dims = Dimensions::new_2d(100, 200);
        assert!(dims.is_2d());
        assert!(!dims.is_3d());
        assert_eq!(dims.effective_shape(), vec![100, 200]);
    }

    #[test]
    fn test_linear_index() {
        let dims = Dimensions::new(2, 3, 4, 10, 20);

        // Test round-trip
        let idx = dims.linear_index(1, 2, 3, 5, 10);
        let (t, c, z, y, x) = dims.coords_from_linear(idx);
        assert_eq!((t, c, z, y, x), (1, 2, 3, 5, 10));
    }

    #[test]
    fn test_dimension_order_parse() {
        assert_eq!(
            DimensionOrder::from_str("TCZYX"),
            Some(DimensionOrder::TCZYX)
        );
        assert_eq!(DimensionOrder::from_str("ZYX"), Some(DimensionOrder::ZYX));
        assert_eq!(DimensionOrder::from_str("YX"), Some(DimensionOrder::YX));
    }
}
