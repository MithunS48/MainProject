"""
Unit tests for helper functions in app.py.

Task 5.1 — validate_file_format
Requirements: 6.4, 1.3
"""

import pytest

# The lifespan context manager only runs when the ASGI app is started (via
# uvicorn or TestClient), NOT on plain module import.  It is therefore safe to
# import the helper functions directly without triggering model loading.
from app import validate_file_format


# ---------------------------------------------------------------------------
# validate_file_format — accepted extensions
# ---------------------------------------------------------------------------


class TestValidateFileFormatAccepted:
    """Filenames with supported extensions should return True."""

    def test_jpg_lowercase(self):
        assert validate_file_format("fish.jpg") is True

    def test_jpeg_lowercase(self):
        assert validate_file_format("fish.jpeg") is True

    def test_png_lowercase(self):
        assert validate_file_format("fish.png") is True

    def test_jpg_uppercase(self):
        assert validate_file_format("fish.JPG") is True

    def test_jpeg_uppercase(self):
        assert validate_file_format("fish.JPEG") is True

    def test_png_uppercase(self):
        assert validate_file_format("fish.PNG") is True

    def test_jpg_mixed_case(self):
        assert validate_file_format("fish.Jpg") is True

    def test_jpeg_mixed_case(self):
        assert validate_file_format("fish.JpEg") is True

    def test_png_mixed_case(self):
        assert validate_file_format("fish.Png") is True

    def test_filename_with_dots_in_stem(self):
        """Dots in the stem should not confuse extension parsing."""
        assert validate_file_format("my.fish.photo.jpg") is True

    def test_filename_with_spaces(self):
        assert validate_file_format("my fish photo.png") is True


# ---------------------------------------------------------------------------
# validate_file_format — rejected extensions
# ---------------------------------------------------------------------------


class TestValidateFileFormatRejected:
    """Filenames with unsupported extensions should return False."""

    def test_gif(self):
        assert validate_file_format("animation.gif") is False

    def test_bmp(self):
        assert validate_file_format("image.bmp") is False

    def test_txt(self):
        assert validate_file_format("notes.txt") is False

    def test_pdf(self):
        assert validate_file_format("document.pdf") is False

    def test_webp(self):
        assert validate_file_format("photo.webp") is False

    def test_tiff(self):
        assert validate_file_format("scan.tiff") is False

    def test_svg(self):
        assert validate_file_format("icon.svg") is False


# ---------------------------------------------------------------------------
# validate_file_format — edge cases
# ---------------------------------------------------------------------------


class TestValidateFileFormatEdgeCases:
    """Edge cases: no extension, empty string, unusual inputs."""

    def test_no_extension(self):
        """A filename with no dot should return False."""
        assert validate_file_format("image") is False

    def test_empty_string(self):
        """An empty string should return False."""
        assert validate_file_format("") is False

    def test_only_dot(self):
        """A filename that is just a dot has no meaningful extension."""
        assert validate_file_format(".") is False

    def test_dot_only_extension(self):
        """A filename ending with a dot (empty extension) should return False."""
        assert validate_file_format("image.") is False

    def test_hidden_file_no_extension(self):
        """A Unix hidden file with no extension (e.g. '.gitignore') should return False."""
        assert validate_file_format(".gitignore") is False

    def test_extension_only(self):
        """A string that is just an extension (e.g. '.jpg') — the stem is empty but
        the extension is valid; behaviour depends on rsplit logic."""
        # rsplit(".", 1) on ".jpg" → ["", "jpg"] → ext = "jpg" → True
        assert validate_file_format(".jpg") is True

    def test_whitespace_only(self):
        """A whitespace-only string has no extension and should return False."""
        assert validate_file_format("   ") is False
