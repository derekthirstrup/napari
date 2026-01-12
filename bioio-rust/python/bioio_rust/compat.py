"""
Compatibility layer for bioio API.

This module provides a drop-in replacement for the bioio BioImage class,
allowing existing code to use bioio-rust with minimal changes.

Example:
    # Instead of:
    # from bioio import BioImage

    # Use:
    from bioio_rust.compat import BioImage

    img = BioImage("image.tiff")
    data = img.data
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type, Union

import numpy as np

from bioio_rust import BioImage as _RustBioImage
from bioio_rust import imread as _rust_imread

if TYPE_CHECKING:
    import dask.array as da
    import xarray as xr


class BioImage:
    """
    bioio-compatible BioImage class backed by Rust implementation.

    This class provides the same interface as bioio.BioImage while using
    the high-performance Rust backend for actual file reading.

    Args:
        image: Path to image file or array-like data
        reader: Optional reader class (for API compatibility, converted to string)
        scene: Scene index to open
        **kwargs: Additional keyword arguments (num_threads, etc.)

    Example:
        >>> from bioio_rust.compat import BioImage
        >>> img = BioImage("experiment.nd2")
        >>> print(img.shape)
        >>> data = img.data
    """

    def __init__(
        self,
        image: Union[str, Path, "np.ndarray"],
        reader: Optional[Type] = None,
        scene: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._path: Optional[str] = None
        self._array_data: Optional[np.ndarray] = None
        self._impl: Optional[_RustBioImage] = None

        # Handle array input
        if isinstance(image, np.ndarray):
            self._array_data = image
            self._path = None
        else:
            # Convert Path to string
            path = str(image) if isinstance(image, Path) else image
            self._path = path

            # Map reader class to string
            reader_str = None
            if reader is not None:
                reader_name = getattr(reader, "__module__", "").split(".")[-1]
                if "tiff" in reader_name.lower():
                    reader_str = "tiff"
                elif "nd2" in reader_name.lower():
                    reader_str = "nd2"
                elif "ome" in reader_name.lower():
                    reader_str = "ome-tiff"
                elif "png" in reader_name.lower():
                    reader_str = "png"

            num_threads = kwargs.get("num_threads", 0)

            self._impl = _RustBioImage(
                path,
                reader=reader_str,
                scene=scene,
                num_threads=num_threads,
            )

    @property
    def shape(self) -> tuple:
        """Image shape as (T, C, Z, Y, X)."""
        if self._array_data is not None:
            return self._array_data.shape
        return tuple(self._impl.shape)

    @property
    def dims(self) -> tuple:
        """Dimension names."""
        if self._impl is not None:
            return tuple(self._impl.dims)
        return ("T", "C", "Z", "Y", "X")[: len(self.shape)]

    @property
    def dtype(self) -> np.dtype:
        """NumPy data type."""
        if self._array_data is not None:
            return self._array_data.dtype
        return np.dtype(self._impl.dtype)

    @property
    def data(self) -> np.ndarray:
        """Load and return image data as numpy array."""
        if self._array_data is not None:
            return self._array_data
        return np.asarray(self._impl.data)

    @property
    def dask_data(self) -> "da.Array":
        """
        Return data wrapped in dask for lazy evaluation.

        Note: The Rust implementation loads eagerly, but wraps in dask
        for API compatibility with bioio.
        """
        try:
            import dask.array as da
        except ImportError:
            raise ImportError(
                "dask is required for dask_data. "
                "Install with: pip install bioio-rust[compat]"
            )

        return da.from_array(self.data, chunks="auto")

    @property
    def xarray_data(self) -> "xr.DataArray":
        """Return data as xarray DataArray."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for xarray_data. "
                "Install with: pip install bioio-rust[compat]"
            )

        return xr.DataArray(
            self.data,
            dims=self.dims,
            name=Path(self._path).stem if self._path else "data",
        )

    @property
    def xarray_dask_data(self) -> "xr.DataArray":
        """Return data as xarray DataArray backed by dask."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for xarray_dask_data. "
                "Install with: pip install bioio-rust[compat]"
            )

        return xr.DataArray(
            self.dask_data,
            dims=self.dims,
            name=Path(self._path).stem if self._path else "data",
        )

    @property
    def scenes(self) -> list[str]:
        """List of scene names."""
        if self._impl is not None:
            return self._impl.scenes
        return ["Scene_0"]

    @property
    def current_scene(self) -> str:
        """Current scene name."""
        if self._impl is not None:
            idx = self._impl.current_scene
            scenes = self.scenes
            return scenes[idx] if idx < len(scenes) else f"Scene_{idx}"
        return "Scene_0"

    @property
    def current_scene_index(self) -> int:
        """Current scene index."""
        if self._impl is not None:
            return self._impl.current_scene
        return 0

    @property
    def channel_names(self) -> list[str]:
        """List of channel names."""
        if self._impl is not None:
            return self._impl.metadata.channel_names
        return [f"Channel_{i}" for i in range(self.shape[1] if len(self.shape) > 1 else 1)]

    @property
    def physical_pixel_sizes(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Physical pixel sizes as (Z, Y, X) in micrometers."""
        if self._impl is not None:
            meta = self._impl.metadata
            return (meta.pixel_size_z, meta.pixel_size_y, meta.pixel_size_x)
        return (None, None, None)

    @property
    def metadata(self) -> Any:
        """Raw metadata object."""
        if self._impl is not None:
            return self._impl.metadata
        return None

    def set_scene(self, scene: Union[int, str]) -> None:
        """
        Set the current scene.

        Note: This creates a new BioImage instance internally.
        """
        if isinstance(scene, str):
            try:
                scene = self.scenes.index(scene)
            except ValueError:
                raise ValueError(f"Scene '{scene}' not found. Available: {self.scenes}")

        if self._path is not None:
            self._impl = _RustBioImage(self._path, scene=scene)

    def get_image_data(
        self,
        dimension_order_out: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Get image data with optional dimension reordering.

        Args:
            dimension_order_out: Output dimension order (e.g., "CZYX")
            **kwargs: Slicing parameters (T, C, Z, Y, X)

        Returns:
            NumPy array with requested dimensions
        """
        data = self.data

        # Apply slicing
        slices = []
        current_dims = list(self.dims)

        for dim in current_dims:
            if dim in kwargs:
                val = kwargs[dim]
                if isinstance(val, int):
                    slices.append(val)
                elif isinstance(val, slice):
                    slices.append(val)
                else:
                    slices.append(slice(None))
            else:
                slices.append(slice(None))

        data = data[tuple(slices)]

        # Reorder dimensions if requested
        if dimension_order_out is not None:
            # Remove dimensions that were indexed with integers
            remaining_dims = [
                d for d, s in zip(current_dims, slices) if not isinstance(s, int)
            ]

            # Calculate transpose order
            target_dims = list(dimension_order_out.upper())
            transpose_order = []
            for target in target_dims:
                if target in remaining_dims:
                    transpose_order.append(remaining_dims.index(target))

            if transpose_order:
                data = np.transpose(data, transpose_order)

        return data

    def get_image_dask_data(
        self,
        dimension_order_out: Optional[str] = None,
        **kwargs: Any,
    ) -> "da.Array":
        """Get image data as dask array with optional dimension reordering."""
        try:
            import dask.array as da
        except ImportError:
            raise ImportError("dask is required for get_image_dask_data")

        data = self.get_image_data(dimension_order_out, **kwargs)
        return da.from_array(data, chunks="auto")

    def __repr__(self) -> str:
        if self._impl is not None:
            return repr(self._impl)
        return f"BioImage(shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        return self.__repr__()


def imread(path: Union[str, Path], **kwargs: Any) -> np.ndarray:
    """
    Read an image file and return as numpy array.

    Args:
        path: Path to image file
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        NumPy array with image data
    """
    return np.asarray(_rust_imread(str(path)))
