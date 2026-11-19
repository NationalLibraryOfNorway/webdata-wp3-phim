from typing import Generator

import numpy as np
import PIL.Image as Image
from phim import compute_phash, compute_phashes


def test_hashes_dihedral_transforms(
    arrow_images: Generator[Image.Image, None, None],
) -> None:
    """
    The hashes should correspond to calculating a hash for each dihedral transform of the image.
    """
    arrow_images = list(arrow_images)
    hashes = compute_phashes(arrow_images[0])

    for transformed_arrow, hash_ in zip(arrow_images, hashes):
        hash_of_transformed_arrow = compute_phash(transformed_arrow)
        np.testing.assert_allclose(hash_, hash_of_transformed_arrow)
