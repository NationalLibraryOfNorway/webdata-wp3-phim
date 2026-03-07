from itertools import product
from typing import Generator

import numpy as np
import PIL.Image as Image
import pytest
from phim import (
    compute_hamming_distance,
    compute_pdq_hashes,
)


def test_same_image_gives_same_hash(sun_image: Image.Image) -> None:
    """Calculating the hash twice for the same image should yield the same result."""
    hashes1, quality1 = compute_pdq_hashes(sun_image)
    hashes2, quality2 = compute_pdq_hashes(sun_image)
    np.testing.assert_allclose(hashes1, hashes2)
    assert quality1 == quality2


def assert_same_hashes_and_quality(
    hashes1, quality1, hashes2, quality2, tolerance
) -> None:
    assert quality1 == quality2
    for h1, h2 in zip(hashes1, hashes2):
        assert compute_hamming_distance(h1, h2) <= tolerance


def test_same_image_different_file_type(
    sun_image: Image.Image,
    sun_images_jpg_compressed: Generator[Image.Image, None, None],
) -> None:
    """Calculating the hash for the same image with different file types should yield approximately the same hash.

    We verify this by comparing the hash of the original image with the hash of the JPEG-compressed version
    (compressed with 100% quality, 85% quality and 50% quality)
    """

    jpg_q100, jpg_q085, jpg_q050, _, _ = sun_images_jpg_compressed
    hash, quality = compute_pdq_hashes(sun_image)

    assert_same_hashes_and_quality(
        hash, quality, *compute_pdq_hashes(jpg_q100), tolerance=8
    )
    assert_same_hashes_and_quality(
        hash, quality, *compute_pdq_hashes(jpg_q085), tolerance=8
    )
    assert_same_hashes_and_quality(
        hash, quality, *compute_pdq_hashes(jpg_q050), tolerance=8
    )


def test_different_image_gives_different_hash(
    sun_image: Image.Image, moon_image: Image.Image
) -> None:
    """Calculating the hash for two different images should yield different results."""
    hashes1 = compute_pdq_hashes(sun_image)[0]
    hashes2 = compute_pdq_hashes(moon_image)[0]
    assert not np.allclose(hashes1, hashes2)


def test_same_images_more_similar_than_different_image(
    sun_image: Image.Image,
    sun_images_jpg_compressed: Generator[Image.Image, None, None],
    moon_images_jpg_compressed: Generator[Image.Image, None, None],
) -> None:
    """Hash of the same image with different compressions should always be closer than different image"""

    hashes = compute_pdq_hashes(sun_image)[0]

    for compressed_sun, compressed_moon in product(
        sun_images_jpg_compressed, moon_images_jpg_compressed
    ):
        d1 = compute_hamming_distance(
            hashes[0], compute_pdq_hashes(compressed_sun)[0][0]
        )
        d2 = compute_hamming_distance(
            hashes[0], compute_pdq_hashes(compressed_moon)[0][0]
        )
        assert d1 < d2


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (100, 100), color="black"),
        Image.new("RGB", (100, 100), color="white"),
    ],
)
def test_quality_zero_for_emtpy_image(image: Image.Image) -> None:
    """The computed quality should be zero for a blank image"""
    _hashes, quality = compute_pdq_hashes(image)
    assert quality == 0


def test_quality_not_zero_for_nonemtpy_image(
    sun_image: Image.Image, moon_image: Image.Image
) -> None:
    """If the image is not blank, quality should be above non-zero"""
    for image in (sun_image, moon_image):
        _hashes, quality = compute_pdq_hashes(image)
        assert quality != 0
