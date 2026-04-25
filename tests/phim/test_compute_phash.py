from collections.abc import Generator
from itertools import product

import numpy as np
import PIL.Image as Image
import pytest
from phim import compute_hamming_distance, compute_phash


def test_same_image_gives_same_hash(sun_image: Image.Image) -> None:
    """Calculating the hash twice for the same image should yield the same result."""
    hash1 = compute_phash(sun_image)
    hash2 = compute_phash(sun_image)
    np.testing.assert_allclose(hash1, hash2)


def test_same_image_different_file_type(
    moon_image: Image.Image,
    moon_images_jpg_compressed: Generator[Image.Image, None, None],
) -> None:
    """Calculating the hash for the same image with different file types should yield the same result.

    We verify this by comparing the hash of the original image with the hash of the JPEG-compressed version
    (compressed with 100% quality, 85% quality and 50% quality)
    """

    jpg_q100, jpg_q085, jpg_q050, _, _ = moon_images_jpg_compressed
    hash_ = compute_phash(moon_image)

    np.testing.assert_allclose(hash_, compute_phash(jpg_q100))
    np.testing.assert_allclose(hash_, compute_phash(jpg_q085))
    np.testing.assert_allclose(hash_, compute_phash(jpg_q050))


def test_different_image_gives_different_hash(sun_image: Image.Image, moon_image: Image.Image) -> None:
    """Calculating the hash for two different images should yield different results."""
    hash1 = compute_phash(sun_image)
    hash2 = compute_phash(moon_image)
    assert not np.allclose(hash1, hash2)


def test_same_images_more_similar_than_different_image(
    sun_image: Image.Image,
    sun_images_jpg_compressed: Generator[Image.Image, None, None],
    moon_images_jpg_compressed: Generator[Image.Image, None, None],
) -> None:
    """Hash of the same image with different compressions should always be closed than different image"""

    hash = compute_phash(sun_image)

    for compressed_sun, compressed_mon in product(sun_images_jpg_compressed, moon_images_jpg_compressed):
        d1 = compute_hamming_distance(hash, compute_phash(compressed_sun))
        d2 = compute_hamming_distance(hash, compute_phash(compressed_mon))
        assert d1 < d2


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_half_different_image_gives_different_hash(
    ocean_image: Image.Image, circles_image: Image.Image, rotation: int
) -> None:
    """Calculating the hash for two different images should yield different results."""
    hash1 = compute_phash(ocean_image.rotate(rotation))
    hash2 = compute_phash(circles_image.rotate(rotation))

    assert not np.allclose(hash1, hash2)


def test_black_and_transparent_gets_nonzero_hash(black_spiral_image: Image.Image) -> None:
    """An all black image on transparent background should get nonzero hash"""
    hash_ = compute_phash(black_spiral_image)
    assert hash_.any()


def test_white_and_transparent_gets_nonzero_hash(white_spiral_image: Image.Image) -> None:
    """An all white image on transparent background should get nonzero hash"""
    hash_ = compute_phash(white_spiral_image)
    assert hash_.any()
