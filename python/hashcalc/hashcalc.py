import numpy as np
from PIL import Image

import hashcalc._native


def compute_phash(img: Image.Image) -> int:
    """Compute the perceptual hash (DCT) of an image.

    https://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html
    """
    rgb_img = img.convert("RGB")

    return hashcalc._native.compute_phash(np.ascontiguousarray(rgb_img))


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()
