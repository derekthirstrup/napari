"""Type stubs for bioio_rust._core"""

from typing import Optional, List, Dict, Any
import numpy as np
from numpy.typing import NDArray

__version__: str

class BioImage:
    """High-performance microscopy image reader.

    Attributes:
        shape: Image dimensions as (T, C, Z, Y, X)
        dims: Dimension names ["T", "C", "Z", "Y", "X"]
        dtype: NumPy dtype string
        format: Image format name
        path: File path
        num_scenes: Number of scenes/series
        scenes: List of scene names
        current_scene: Current scene index
        metadata: Image metadata
        data: Image data as NumPy array
    """

    def __init__(
        self,
        path: str,
        reader: Optional[str] = None,
        scene: Optional[int] = None,
        num_threads: int = 0,
    ) -> None:
        """Open an image file.

        Args:
            path: Path to image file
            reader: Optional reader name ("tiff", "nd2", "ome-tiff", "png")
            scene: Optional scene index to open
            num_threads: Number of threads for parallel reading (0 = auto)
        """
        ...

    @property
    def shape(self) -> List[int]: ...

    @property
    def dims(self) -> List[str]: ...

    @property
    def dtype(self) -> str: ...

    @property
    def format(self) -> str: ...

    @property
    def path(self) -> str: ...

    @property
    def num_scenes(self) -> int: ...

    @property
    def scenes(self) -> List[str]: ...

    @property
    def current_scene(self) -> int: ...

    @property
    def metadata(self) -> "Metadata": ...

    @property
    def data(self) -> NDArray[np.uint8]:
        """Read all image data as NumPy array.

        Returns:
            NumPy array with shape (T, C, Z, Y, X)
        """
        ...

    def read_parallel(self, num_threads: int = 0) -> NDArray[np.uint8]:
        """Read data with explicit thread count.

        Args:
            num_threads: Number of threads (0 = auto)

        Returns:
            NumPy array with shape (T, C, Z, Y, X)
        """
        ...

    def set_scene(self, scene: int) -> None:
        """Set the current scene."""
        ...


class Metadata:
    """Image metadata container.

    Attributes:
        pixel_size_x: Pixel size in X (micrometers)
        pixel_size_y: Pixel size in Y (micrometers)
        pixel_size_z: Pixel size in Z (micrometers)
        channel_names: List of channel names
        image_name: Image name
        acquisition_date: Acquisition date string
        software: Software used to create image
        description: Image description
    """

    @property
    def pixel_size_x(self) -> Optional[float]: ...

    @property
    def pixel_size_y(self) -> Optional[float]: ...

    @property
    def pixel_size_z(self) -> Optional[float]: ...

    @property
    def channel_names(self) -> List[str]: ...

    @property
    def image_name(self) -> Optional[str]: ...

    @property
    def acquisition_date(self) -> Optional[str]: ...

    @property
    def software(self) -> Optional[str]: ...

    @property
    def description(self) -> Optional[str]: ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        ...


class Dimensions:
    """Image dimension information.

    Attributes:
        t: Number of timepoints
        c: Number of channels
        z: Number of Z slices
        y: Height in pixels
        x: Width in pixels
        shape: Dimensions as list [T, C, Z, Y, X]
    """

    @property
    def t(self) -> int: ...

    @property
    def c(self) -> int: ...

    @property
    def z(self) -> int: ...

    @property
    def y(self) -> int: ...

    @property
    def x(self) -> int: ...

    @property
    def shape(self) -> List[int]: ...


def imread(
    path: str,
    reader: Optional[str] = None,
) -> NDArray[np.uint8]:
    """Read image file as NumPy array.

    This is the fastest way to read a single image.

    Args:
        path: Path to image file
        reader: Optional reader name

    Returns:
        NumPy array with shape (T, C, Z, Y, X)
    """
    ...


def imread_batch(
    paths: List[str],
    num_threads: int = 0,
) -> List[NDArray[np.uint8]]:
    """Read multiple image files in parallel.

    Args:
        paths: List of paths to image files
        num_threads: Number of threads (0 = auto)

    Returns:
        List of NumPy arrays
    """
    ...


def imwrite(
    path: str,
    data: NDArray[np.uint8],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write image data to file.

    Args:
        path: Output file path
        data: NumPy array with image data
        metadata: Optional metadata dictionary
    """
    ...


def detect_format(path: str) -> str:
    """Detect image format from file.

    Args:
        path: Path to image file

    Returns:
        Format name string
    """
    ...


def get_supported_formats() -> List[str]:
    """Get list of supported format names."""
    ...
