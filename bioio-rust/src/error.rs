//! Error types for bioio-rust

use thiserror::Error;

/// Main error type for bioio-rust operations
#[derive(Error, Debug)]
pub enum BioIoError {
    /// I/O error from file operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Format not supported
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Invalid TIFF file
    #[error("Invalid TIFF: {0}")]
    InvalidTiff(String),

    /// Invalid ND2 file
    #[error("Invalid ND2: {0}")]
    InvalidNd2(String),

    /// Invalid OME-TIFF file
    #[error("Invalid OME-TIFF: {0}")]
    InvalidOmeTiff(String),

    /// Invalid PNG file
    #[error("Invalid PNG: {0}")]
    InvalidPng(String),

    /// Decompression error
    #[error("Decompression error: {0}")]
    Decompression(String),

    /// Invalid dimensions
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// XML parsing error
    #[error("XML parsing error: {0}")]
    XmlParse(String),

    /// Scene not found
    #[error("Scene {0} not found (available: {1})")]
    SceneNotFound(usize, usize),

    /// Invalid region
    #[error("Invalid region: {0}")]
    InvalidRegion(String),

    /// Thread pool error
    #[error("Thread pool error: {0}")]
    ThreadPool(String),

    /// Data type mismatch
    #[error("Data type mismatch: expected {expected}, got {got}")]
    DTypeMismatch { expected: String, got: String },

    /// Python error
    #[error("Python error: {0}")]
    Python(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// Result type alias for bioio operations
pub type Result<T> = std::result::Result<T, BioIoError>;

// Convert to PyO3 error
impl From<BioIoError> for pyo3::PyErr {
    fn from(err: BioIoError) -> pyo3::PyErr {
        match &err {
            BioIoError::Io(_) => {
                pyo3::exceptions::PyIOError::new_err(err.to_string())
            }
            BioIoError::UnsupportedFormat(_) => {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            }
            BioIoError::InvalidTiff(_)
            | BioIoError::InvalidNd2(_)
            | BioIoError::InvalidOmeTiff(_)
            | BioIoError::InvalidPng(_) => {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            }
            BioIoError::SceneNotFound(_, _) => {
                pyo3::exceptions::PyIndexError::new_err(err.to_string())
            }
            _ => pyo3::exceptions::PyRuntimeError::new_err(err.to_string()),
        }
    }
}

// Conversions from other error types
impl From<quick_xml::Error> for BioIoError {
    fn from(err: quick_xml::Error) -> Self {
        BioIoError::XmlParse(err.to_string())
    }
}

impl From<roxmltree::Error> for BioIoError {
    fn from(err: roxmltree::Error) -> Self {
        BioIoError::XmlParse(err.to_string())
    }
}

impl From<serde_json::Error> for BioIoError {
    fn from(err: serde_json::Error) -> Self {
        BioIoError::Other(format!("JSON error: {}", err))
    }
}

impl From<rayon::ThreadPoolBuildError> for BioIoError {
    fn from(err: rayon::ThreadPoolBuildError) -> Self {
        BioIoError::ThreadPool(err.to_string())
    }
}

impl From<tiff::TiffError> for BioIoError {
    fn from(err: tiff::TiffError) -> Self {
        BioIoError::InvalidTiff(err.to_string())
    }
}
