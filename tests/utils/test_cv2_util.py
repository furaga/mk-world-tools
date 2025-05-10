import pytest
import cv2
import numpy as np
import tempfile
from pathlib import Path

# Assuming src is in PYTHONPATH or using a proper project structure
from utils.cv2_util import imread_safe, imwrite_safe


# Helper function to create a dummy image
def create_dummy_image(width=64, height=64, channels=3, dtype=np.uint8):
    if channels == 1:
        return np.random.randint(0, 256, (height, width), dtype=dtype)
    return np.random.randint(0, 256, (height, width, channels), dtype=dtype)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


def test_imread_safe_valid_image_png(temp_dir):
    img_orig = create_dummy_image()
    file_path = temp_dir / "test_image.png"
    cv2.imwrite(str(file_path), img_orig)  # Use standard cv2.imwrite for setup

    img_read = imread_safe(str(file_path))
    assert img_read is not None
    assert np.array_equal(img_read, img_orig)


def test_imread_safe_valid_image_jpg(temp_dir):
    img_orig = create_dummy_image()
    file_path = temp_dir / "test_image.jpg"
    cv2.imwrite(str(file_path), img_orig)

    img_read = imread_safe(str(file_path))
    assert img_read is not None
    # JPEG is lossy, so we can't do an exact match, check shape and type
    assert img_read.shape == img_orig.shape
    assert img_read.dtype == img_orig.dtype


def test_imread_safe_japanese_filename(temp_dir):
    img_orig = create_dummy_image()
    file_path = temp_dir / "テスト画像.png"
    imwrite_safe(str(file_path), img_orig)

    img_read = imread_safe(str(file_path))
    assert img_read is not None
    assert np.array_equal(img_read, img_orig)


def test_imread_safe_non_existent_file(caplog):
    file_path = "non_existent_image.png"
    img_read = imread_safe(file_path)
    assert img_read is None
    assert f"Failed to read image file: {file_path}" in caplog.text


def test_imread_safe_corrupted_file(temp_dir, caplog):
    file_path = temp_dir / "corrupted.png"
    with open(file_path, "w") as f:
        f.write("this is not an image")

    img_read = imread_safe(str(file_path))
    assert img_read is None


# --- Tests for imwrite_safe ---


def test_imwrite_safe_valid_image_png(temp_dir):
    img_orig = create_dummy_image()
    file_path = temp_dir / "output_image.png"

    result = imwrite_safe(str(file_path), img_orig)
    assert result is True
    assert file_path.exists()

    img_read = imread_safe(str(file_path))  # Use our own imread_safe to verify
    assert img_read is not None
    assert np.array_equal(img_read, img_orig)


def test_imwrite_safe_valid_image_jpg(temp_dir):
    img_orig = create_dummy_image()
    file_path = temp_dir / "output_image.jpg"

    result = imwrite_safe(str(file_path), img_orig)
    assert result is True
    assert file_path.exists()

    img_read = imread_safe(str(file_path))
    assert img_read is not None
    assert img_read.shape == img_orig.shape
    assert img_read.dtype == img_orig.dtype


def test_imwrite_safe_japanese_filename(temp_dir):
    img_orig = create_dummy_image()
    file_path = temp_dir / "出力画像名.png"

    result = imwrite_safe(str(file_path), img_orig)
    assert result is True
    assert file_path.exists()

    img_read = imread_safe(str(file_path))
    assert img_read is not None
    assert np.array_equal(img_read, img_orig)


def test_imwrite_safe_fail_to_encode(temp_dir, mocker, caplog):
    # Test case where cv2.imencode fails
    img_orig = create_dummy_image()
    file_path = temp_dir / "fail_encode.png"

    # Mock cv2.imencode to simulate failure
    mocker.patch("cv2.imencode", return_value=(False, None))

    result = imwrite_safe(str(file_path), img_orig)
    assert result is False
    assert not file_path.exists()


def test_imwrite_safe_fail_to_write_file(temp_dir, mocker, caplog):
    img_orig = create_dummy_image()
    file_path = temp_dir / "cant_write.png"

    # Mock open to simulate a file writing failure
    mocker.patch("builtins.open", side_effect=IOError("Permission denied"))

    result = imwrite_safe(str(file_path), img_orig)
    assert result is False
    assert f"Failed to write image file: {str(file_path)}" in caplog.text
    assert "Permission denied" in caplog.text  # Check for the specific IOError message
