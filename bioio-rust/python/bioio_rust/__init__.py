"""
BioIO Rust - High-performance microscopy image I/O

This package provides fast, parallel image reading for microscopy formats
with zero-copy NumPy integration, powered by Rust.

Example:
    >>> from bioio_rust import BioImage, imread
    >>>
    >>> # Class-based API
    >>> img = BioImage("image.tiff")
    >>> data = img.data  # Returns NumPy array
    >>> print(img.shape)  # (T, C, Z, Y, X)
    >>>
    >>> # Function API (faster for one-off reads)
    >>> data = imread("image.tiff")
    >>>
    >>> # Batch parallel reading
    >>> from bioio_rust import imread_batch
    >>> arrays = imread_batch(["img1.tiff", "img2.tiff"], num_threads=8)

Supported formats:
    - TIFF/BigTIFF
    - OME-TIFF
    - Nikon ND2
    - PNG
"""

from bioio_rust._core import (
    BioImage,
    Metadata,
    Dimensions,
    imread,
    imread_batch,
    imwrite,
    detect_format,
    get_supported_formats,
    __version__,
)

__all__ = [
    "BioImage",
    "Metadata",
    "Dimensions",
    "imread",
    "imread_batch",
    "imwrite",
    "detect_format",
    "get_supported_formats",
    "__version__",
]
