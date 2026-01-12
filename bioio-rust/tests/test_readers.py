"""Tests for bioio-rust readers."""

import numpy as np
import pytest
import tempfile
from pathlib import Path


def create_test_tiff(path: Path, shape: tuple = (100, 100), dtype: np.dtype = np.uint8):
    """Create a minimal test TIFF file."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not installed")

    data = np.random.randint(0, 255, shape, dtype=dtype)
    tifffile.imwrite(str(path), data)
    return data


def create_test_png(path: Path, shape: tuple = (100, 100)):
    """Create a test PNG file."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not installed")

    data = np.random.randint(0, 255, shape, dtype=np.uint8)
    img = Image.fromarray(data)
    img.save(str(path))
    return data


class TestBioImage:
    """Test BioImage class."""

    def test_import(self):
        """Test that bioio_rust can be imported."""
        from bioio_rust import BioImage, imread, detect_format
        assert BioImage is not None
        assert imread is not None
        assert detect_format is not None

    def test_read_tiff(self, tmp_path):
        """Test reading a TIFF file."""
        from bioio_rust import BioImage

        tiff_path = tmp_path / "test.tiff"
        expected = create_test_tiff(tiff_path, (100, 100))

        img = BioImage(str(tiff_path))

        assert img.shape[-2:] == (100, 100)
        assert img.dtype == "uint8"
        assert img.format in ("Tiff", "TIFF")

        data = img.data
        assert data.shape[-2:] == (100, 100)

    def test_read_tiff_16bit(self, tmp_path):
        """Test reading a 16-bit TIFF file."""
        from bioio_rust import BioImage

        tiff_path = tmp_path / "test16.tiff"
        create_test_tiff(tiff_path, (50, 50), dtype=np.uint16)

        img = BioImage(str(tiff_path))
        assert img.dtype in ("uint8", "uint16")

    def test_read_multipage_tiff(self, tmp_path):
        """Test reading a multi-page TIFF file."""
        from bioio_rust import BioImage

        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")

        tiff_path = tmp_path / "multipage.tiff"
        data = np.random.randint(0, 255, (5, 100, 100), dtype=np.uint8)
        tifffile.imwrite(str(tiff_path), data)

        img = BioImage(str(tiff_path))
        # Should have multiple Z slices
        assert img.shape[2] == 5 or np.prod(img.shape) == np.prod(data.shape)

    def test_read_png(self, tmp_path):
        """Test reading a PNG file."""
        from bioio_rust import BioImage

        png_path = tmp_path / "test.png"
        create_test_png(png_path, (80, 80))

        img = BioImage(str(png_path))
        assert img.shape[-2:] == (80, 80)
        assert img.format in ("Png", "PNG")

    def test_metadata(self, tmp_path):
        """Test metadata access."""
        from bioio_rust import BioImage

        tiff_path = tmp_path / "test.tiff"
        create_test_tiff(tiff_path)

        img = BioImage(str(tiff_path))
        meta = img.metadata

        assert meta is not None
        assert hasattr(meta, "pixel_size_x")
        assert hasattr(meta, "channel_names")

    def test_scenes(self, tmp_path):
        """Test scene access."""
        from bioio_rust import BioImage

        tiff_path = tmp_path / "test.tiff"
        create_test_tiff(tiff_path)

        img = BioImage(str(tiff_path))

        assert img.num_scenes >= 1
        assert len(img.scenes) >= 1
        assert img.current_scene >= 0

    def test_explicit_reader(self, tmp_path):
        """Test specifying reader explicitly."""
        from bioio_rust import BioImage

        tiff_path = tmp_path / "test.tiff"
        create_test_tiff(tiff_path)

        img = BioImage(str(tiff_path), reader="tiff")
        assert img.format in ("Tiff", "TIFF")

    def test_parallel_read(self, tmp_path):
        """Test parallel reading."""
        from bioio_rust import BioImage

        tiff_path = tmp_path / "test.tiff"
        create_test_tiff(tiff_path, (200, 200))

        img = BioImage(str(tiff_path))
        data = img.read_parallel(num_threads=2)

        assert data.shape[-2:] == (200, 200)


class TestImread:
    """Test imread function."""

    def test_imread_tiff(self, tmp_path):
        """Test imread with TIFF file."""
        from bioio_rust import imread

        tiff_path = tmp_path / "test.tiff"
        create_test_tiff(tiff_path, (100, 100))

        data = imread(str(tiff_path))

        assert isinstance(data, np.ndarray)
        assert data.shape[-2:] == (100, 100)

    def test_imread_png(self, tmp_path):
        """Test imread with PNG file."""
        from bioio_rust import imread

        png_path = tmp_path / "test.png"
        create_test_png(png_path, (50, 50))

        data = imread(str(png_path))

        assert isinstance(data, np.ndarray)
        assert data.shape[-2:] == (50, 50)


class TestImreadBatch:
    """Test batch reading."""

    def test_imread_batch(self, tmp_path):
        """Test batch reading multiple files."""
        from bioio_rust import imread_batch

        # Create multiple test files
        paths = []
        for i in range(3):
            path = tmp_path / f"test_{i}.tiff"
            create_test_tiff(path, (50, 50))
            paths.append(str(path))

        arrays = imread_batch(paths)

        assert len(arrays) == 3
        for arr in arrays:
            assert arr.shape[-2:] == (50, 50)

    def test_imread_batch_parallel(self, tmp_path):
        """Test batch reading with explicit thread count."""
        from bioio_rust import imread_batch

        paths = []
        for i in range(5):
            path = tmp_path / f"test_{i}.tiff"
            create_test_tiff(path, (30, 30))
            paths.append(str(path))

        arrays = imread_batch(paths, num_threads=2)

        assert len(arrays) == 5


class TestDetectFormat:
    """Test format detection."""

    def test_detect_tiff(self, tmp_path):
        """Test TIFF format detection."""
        from bioio_rust import detect_format

        tiff_path = tmp_path / "test.tiff"
        create_test_tiff(tiff_path)

        fmt = detect_format(str(tiff_path))
        assert "TIFF" in fmt.upper() or "Tiff" in fmt

    def test_detect_png(self, tmp_path):
        """Test PNG format detection."""
        from bioio_rust import detect_format

        png_path = tmp_path / "test.png"
        create_test_png(png_path)

        fmt = detect_format(str(png_path))
        assert "PNG" in fmt.upper() or "Png" in fmt


class TestSupportedFormats:
    """Test supported formats listing."""

    def test_get_supported_formats(self):
        """Test getting list of supported formats."""
        from bioio_rust import get_supported_formats

        formats = get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert "TIFF" in formats or "Tiff" in formats


class TestCompat:
    """Test compatibility layer."""

    def test_compat_bioimage(self, tmp_path):
        """Test compatibility BioImage class."""
        from bioio_rust.compat import BioImage

        tiff_path = tmp_path / "test.tiff"
        create_test_tiff(tiff_path, (100, 100))

        img = BioImage(str(tiff_path))

        assert img.shape[-2:] == (100, 100)
        assert img.dims[-2:] == ("Y", "X")
        assert len(img.scenes) >= 1
        assert img.channel_names is not None

    def test_compat_imread(self, tmp_path):
        """Test compatibility imread function."""
        from bioio_rust.compat import imread

        tiff_path = tmp_path / "test.tiff"
        create_test_tiff(tiff_path, (100, 100))

        data = imread(str(tiff_path))

        assert isinstance(data, np.ndarray)
        assert data.shape[-2:] == (100, 100)

    def test_compat_physical_sizes(self, tmp_path):
        """Test physical pixel size access."""
        from bioio_rust.compat import BioImage

        tiff_path = tmp_path / "test.tiff"
        create_test_tiff(tiff_path)

        img = BioImage(str(tiff_path))
        sizes = img.physical_pixel_sizes

        assert len(sizes) == 3  # Z, Y, X


class TestImwrite:
    """Test image writing."""

    def test_imwrite_basic(self, tmp_path):
        """Test basic image writing."""
        from bioio_rust import imwrite, imread

        data = np.random.randint(0, 255, (1, 1, 1, 100, 100), dtype=np.uint8)
        out_path = tmp_path / "output.tiff"

        imwrite(str(out_path), data)

        assert out_path.exists()

        # Read back and verify
        read_data = imread(str(out_path))
        assert read_data.shape[-2:] == (100, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
