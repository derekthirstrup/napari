//! Python bindings for bioio-rust
//!
//! This module provides PyO3-based Python bindings that expose
//! the Rust image readers to Python with zero-copy NumPy integration.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray, PyArrayDyn, ToPyArray, PyArrayMethods};
use ndarray::ArrayD;
use std::path::Path;

use crate::error::{BioIoError, Result};
use crate::format::ImageFormat;
use crate::metadata::Metadata;
use crate::dimensions::Dimensions;
use crate::readers::{Reader, ReaderOptions, DType};
use crate::readers::tiff::TiffReader;
use crate::readers::nd2::Nd2Reader;
use crate::readers::ome_tiff::OmeTiffReader;
use crate::readers::png::PngReader;

/// Python-exposed BioImage class
#[pyclass(name = "BioImage")]
pub struct PyBioImage {
    reader: Box<dyn Reader>,
    path: String,
}

#[pymethods]
impl PyBioImage {
    /// Create a new BioImage from a file path
    ///
    /// Args:
    ///     path: Path to the image file
    ///     reader: Optional reader name ("tiff", "nd2", "ome-tiff", "png")
    ///     scene: Optional scene index to open
    ///     num_threads: Number of threads for parallel reading (0 = auto)
    #[new]
    #[pyo3(signature = (path, reader=None, scene=None, num_threads=0))]
    fn new(
        path: &str,
        reader: Option<&str>,
        scene: Option<usize>,
        num_threads: usize,
    ) -> PyResult<Self> {
        let path_obj = Path::new(path);
        let options = ReaderOptions {
            scene,
            num_threads,
            ..Default::default()
        };

        // Determine format from explicit reader or auto-detect
        let format = if let Some(reader_name) = reader {
            match reader_name.to_lowercase().as_str() {
                "tiff" | "tifffile" => ImageFormat::Tiff,
                "bigtiff" => ImageFormat::BigTiff,
                "nd2" => ImageFormat::Nd2,
                "ome-tiff" | "ome_tiff" | "ometiff" => ImageFormat::OmeTiff,
                "png" => ImageFormat::Png,
                _ => ImageFormat::detect(path_obj)?,
            }
        } else {
            ImageFormat::detect(path_obj)?
        };

        let reader: Box<dyn Reader> = match format {
            ImageFormat::Tiff | ImageFormat::BigTiff => {
                Box::new(TiffReader::open(path_obj, &options)?)
            }
            ImageFormat::OmeTiff => {
                Box::new(OmeTiffReader::open(path_obj, &options)?)
            }
            ImageFormat::Nd2 => {
                Box::new(Nd2Reader::open(path_obj, &options)?)
            }
            ImageFormat::Png => {
                Box::new(PngReader::open(path_obj, &options)?)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported format: {:?}",
                    format
                )));
            }
        };

        Ok(Self {
            reader,
            path: path.to_string(),
        })
    }

    /// Get image shape as tuple (T, C, Z, Y, X)
    #[getter]
    fn shape(&self) -> Vec<usize> {
        let dims = self.reader.dimensions();
        vec![dims.t, dims.c, dims.z, dims.y, dims.x]
    }

    /// Get dimension names
    #[getter]
    fn dims(&self) -> Vec<&'static str> {
        vec!["T", "C", "Z", "Y", "X"]
    }

    /// Get the data type as string
    #[getter]
    fn dtype(&self) -> &'static str {
        self.reader.dtype().numpy_str()
    }

    /// Get the image format
    #[getter]
    fn format(&self) -> String {
        format!("{:?}", self.reader.format())
    }

    /// Get the file path
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    /// Get number of scenes
    #[getter]
    fn num_scenes(&self) -> usize {
        self.reader.num_scenes()
    }

    /// Get scene names
    #[getter]
    fn scenes(&self) -> Vec<String> {
        self.reader.scene_names()
    }

    /// Get current scene index
    #[getter]
    fn current_scene(&self) -> usize {
        self.reader.current_scene()
    }

    /// Get metadata as PyMetadata object
    #[getter]
    fn metadata(&self) -> PyMetadata {
        PyMetadata {
            inner: self.reader.metadata().clone(),
        }
    }

    /// Read all image data as NumPy array
    ///
    /// This releases the GIL during reading for better concurrency.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
        // Release GIL during the heavy lifting
        let array: ArrayD<u8> = py.allow_threads(|| self.reader.read_all())?;

        // Convert to NumPy array
        Ok(array.to_pyarray(py))
    }

    /// Read data with explicit thread count
    ///
    /// Args:
    ///     num_threads: Number of threads (0 = auto)
    #[pyo3(signature = (num_threads=0))]
    fn read_parallel<'py>(
        &self,
        py: Python<'py>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
        let array: ArrayD<u8> = py.allow_threads(|| self.reader.read_parallel(num_threads))?;
        Ok(array.to_pyarray(py))
    }

    /// Set the current scene
    fn set_scene(&mut self, scene: usize) -> PyResult<()> {
        // Note: This requires interior mutability pattern in real implementation
        // For now, we'll return an error suggesting to reopen with scene parameter
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Use BioImage(path, scene=N) to open a specific scene",
        ))
    }

    fn __repr__(&self) -> String {
        let dims = self.reader.dimensions();
        format!(
            "BioImage('{}', shape=({}, {}, {}, {}, {}), dtype='{}', format='{}')",
            self.path,
            dims.t,
            dims.c,
            dims.z,
            dims.y,
            dims.x,
            self.reader.dtype().numpy_str(),
            self.reader.format().name()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python-exposed Metadata class
#[pyclass(name = "Metadata")]
#[derive(Clone)]
pub struct PyMetadata {
    inner: Metadata,
}

#[pymethods]
impl PyMetadata {
    /// Get pixel size in X dimension (micrometers)
    #[getter]
    fn pixel_size_x(&self) -> Option<f64> {
        self.inner.pixel_size_x
    }

    /// Get pixel size in Y dimension (micrometers)
    #[getter]
    fn pixel_size_y(&self) -> Option<f64> {
        self.inner.pixel_size_y
    }

    /// Get pixel size in Z dimension (micrometers)
    #[getter]
    fn pixel_size_z(&self) -> Option<f64> {
        self.inner.pixel_size_z
    }

    /// Get channel names
    #[getter]
    fn channel_names(&self) -> Vec<String> {
        self.inner.channel_names.clone()
    }

    /// Get image name
    #[getter]
    fn image_name(&self) -> Option<String> {
        self.inner.image_name.clone()
    }

    /// Get acquisition date
    #[getter]
    fn acquisition_date(&self) -> Option<String> {
        self.inner.acquisition_date.clone()
    }

    /// Get software name
    #[getter]
    fn software(&self) -> Option<String> {
        self.inner.software.clone()
    }

    /// Get description
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    /// Convert to dictionary
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        if let Some(x) = self.inner.pixel_size_x {
            dict.set_item("pixel_size_x", x)?;
        }
        if let Some(y) = self.inner.pixel_size_y {
            dict.set_item("pixel_size_y", y)?;
        }
        if let Some(z) = self.inner.pixel_size_z {
            dict.set_item("pixel_size_z", z)?;
        }
        dict.set_item("channel_names", &self.inner.channel_names)?;

        if let Some(name) = &self.inner.image_name {
            dict.set_item("image_name", name)?;
        }
        if let Some(date) = &self.inner.acquisition_date {
            dict.set_item("acquisition_date", date)?;
        }
        if let Some(software) = &self.inner.software {
            dict.set_item("software", software)?;
        }

        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Python-exposed Dimensions class
#[pyclass(name = "Dimensions")]
#[derive(Clone)]
pub struct PyDimensions {
    inner: Dimensions,
}

#[pymethods]
impl PyDimensions {
    #[getter]
    fn t(&self) -> usize {
        self.inner.t
    }

    #[getter]
    fn c(&self) -> usize {
        self.inner.c
    }

    #[getter]
    fn z(&self) -> usize {
        self.inner.z
    }

    #[getter]
    fn y(&self) -> usize {
        self.inner.y
    }

    #[getter]
    fn x(&self) -> usize {
        self.inner.x
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape_vec()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Fast imread function - read image directly as NumPy array
///
/// This bypasses class instantiation for maximum performance.
///
/// Args:
///     path: Path to image file
///     reader: Optional reader name
///
/// Returns:
///     NumPy array with shape (T, C, Z, Y, X)
#[pyfunction]
#[pyo3(signature = (path, reader=None))]
pub fn imread<'py>(
    py: Python<'py>,
    path: &str,
    reader: Option<&str>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let img = PyBioImage::new(path, reader, None, 0)?;
    img.data(py)
}

/// Batch read multiple files in parallel
///
/// Args:
///     paths: List of paths to image files
///     num_threads: Number of threads (0 = auto)
///
/// Returns:
///     List of NumPy arrays
#[pyfunction]
#[pyo3(signature = (paths, num_threads=0))]
pub fn imread_batch<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    num_threads: usize,
) -> PyResult<Bound<'py, PyList>> {
    use rayon::prelude::*;

    // Configure thread pool
    let pool = if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
    } else {
        rayon::ThreadPoolBuilder::new()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
    };

    // Read all files in parallel (GIL released)
    let arrays: Vec<Result<ArrayD<u8>>> = py.allow_threads(|| {
        pool.install(|| {
            paths
                .par_iter()
                .map(|path| {
                    let path_obj = Path::new(path);
                    let format = ImageFormat::detect(path_obj)?;
                    let options = ReaderOptions::default();

                    let reader: Box<dyn Reader> = match format {
                        ImageFormat::Tiff | ImageFormat::BigTiff => {
                            Box::new(TiffReader::open(path_obj, &options)?)
                        }
                        ImageFormat::OmeTiff => {
                            Box::new(OmeTiffReader::open(path_obj, &options)?)
                        }
                        ImageFormat::Nd2 => {
                            Box::new(Nd2Reader::open(path_obj, &options)?)
                        }
                        ImageFormat::Png => {
                            Box::new(PngReader::open(path_obj, &options)?)
                        }
                        _ => {
                            return Err(BioIoError::UnsupportedFormat(format!(
                                "{:?}",
                                format
                            )));
                        }
                    };

                    reader.read_all()
                })
                .collect()
        })
    });

    // Convert to Python list of NumPy arrays
    let list = PyList::empty(py);
    for result in arrays {
        let arr = result?;
        list.append(arr.to_pyarray(py))?;
    }

    Ok(list)
}

/// Write an image to file
///
/// Args:
///     path: Output file path
///     data: NumPy array with image data
///     metadata: Optional metadata dictionary
#[pyfunction]
#[pyo3(signature = (path, data, metadata=None))]
pub fn imwrite(
    path: &str,
    data: &Bound<'_, PyArrayDyn<u8>>,
    metadata: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    use crate::writers::{self, WriterOptions};

    // Convert NumPy array to ndarray
    let array = unsafe { data.as_array() };
    let owned = array.to_owned();

    // Determine dimensions from array shape
    let shape = owned.shape();
    let dimensions = match shape.len() {
        2 => Dimensions::new(1, 1, 1, shape[0], shape[1]),
        3 => Dimensions::new(1, shape[0], 1, shape[1], shape[2]),
        4 => Dimensions::new(1, shape[0], shape[1], shape[2], shape[3]),
        5 => Dimensions::new(shape[0], shape[1], shape[2], shape[3], shape[4]),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Array must have 2-5 dimensions",
            ));
        }
    };

    // Convert metadata if provided
    let meta = if let Some(_dict) = metadata {
        // TODO: Parse metadata from dict
        None
    } else {
        None
    };

    let path_obj = Path::new(path);
    writers::imwrite(path_obj, &owned, &dimensions, meta.as_ref(), None)?;

    Ok(())
}

/// Detect image format from file
///
/// Args:
///     path: Path to image file
///
/// Returns:
///     Format name string
#[pyfunction]
pub fn detect_format(path: &str) -> PyResult<String> {
    let format = ImageFormat::detect(Path::new(path))?;
    Ok(format.name().to_string())
}

/// Get list of supported formats
#[pyfunction]
pub fn get_supported_formats() -> Vec<String> {
    ImageFormat::supported_formats()
        .into_iter()
        .map(|f| f.name().to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_formats() {
        let formats = get_supported_formats();
        assert!(formats.contains(&"TIFF".to_string()));
        assert!(formats.contains(&"ND2".to_string()));
    }
}
