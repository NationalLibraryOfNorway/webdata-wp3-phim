"""Utilities for comparing images with perceptual hashes.

All functions wrap implementations in low-level languages (Rust for pHash and hamming distances, C++ for PDQ-hash).
"""

from enum import Enum

import numpy as np
import pdqhash
from numpy.typing import NDArray
from PIL import Image

import hashcalc._native

__all__ = [
    "PDQHashOrder",
    "compute_pdq_hashes",
    "compute_p_hash",
    "compute_hamming_distances",
    "compute_hamming_distance",
]


class PDQHashOrder(Enum):
    """Used to index the PDQ hash iterable to filter based on image transform."""

    ORIGINAL = 0
    ROTATED_90 = 1
    ROTATED_180 = 2
    ROTATED_270 = 3
    FLIPPED_VERTICALLY = 4
    FLIPPED_HORIZONTALLY = 5
    ROTATED_90_FLIPPED_VERTICALLY = 6
    ROTATED_90_FLIPPED_HORIZONTALLY = 7


def compute_pdq_hashes(img: Image.Image) -> tuple[NDArray[np.uint8], int]:
    """Compute eight PDQ-hashes for various transformations of the input image and the image quality.

    The PDQ-hash is a perceptual hash for images developed by Meta for image fingerprinting.
    It is very similar to the pHash algorithm, but it uses more bits per hash (256 bits instead of 64 bits).
    Consequently, two images must be very very similar to have an equal PDQ-hash.

    A benefit with the PDQ-hash is that it is (also) based on the cosine transform, and the reference defines eight
    transforms you get "for free" based on a single cosine transform: All four 90 degree rotations of an image with
    horisontal and vertical flips.
    You could, strictly speaking, get the same transforms for free with pHash, but most implementations do not utilise
    these trics.

    The PDQ-hash also comes with a "quality" metric between 0 and 100.
    This metric is based on the sum of gradient magnitudes: more gradients mean higher quality.
    This is a very simple metric, that can be used to filter out featureless images (and frames from videos).

    This is a simple wrapper around the pdqhash library on PyPI, which wraps the reference C++ implementation.
    The main difference is that this function returns a uint8-array with packed bits instead of a boolean array.
    This is useful because it can be used as input for the :func:`compute_hamming_distance` and
    :func:`compute_hamming_distances` functions.

    For more information about the PDQ-hash, see the reference implementation by Facebook Threat Exchange:
    https://github.com/facebook/ThreatExchange/tree/main/pdq

    Parameters
    ----------
    img
        Pillow image we want to compute the PDQ-hash for

    Returns
    -------
    NDArray[np.uint8]
        An 8x32 numpy array. The first axis corresponds to image transforms, and can be indexed with the
        :class:`PDQHashOrder` enum and the second axis is the hash-axis.

    int
        The image quality metric (0 <= q <= 100)
    """
    hash_vectors, quality = pdqhash.compute_dihedral(np.array(img))
    return np.packbits(hash_vectors, axis=1), quality


def compute_phash(img: Image.Image) -> int:
    """Compute the pHash (DCT) of an image.

    The pHash is based on the discrete cosine transform (DCT) of the image. In particular, the image is resized to
    32 x 32 pixels.
    Then, we compute the DCT of the image and get the 64 lowest frequencies (8-by-8 pixel square).
    Finally, we find the average value of these frequencies, and threshold the DCT matrix based on it to get a binary
    pattern, which constitutes our 64-bit hash.

    For more information, see Chapter 3.2.1 of Christoph Zauner's master's thesis:
    https://www.phash.org/docs/pubs/thesis_zauner.pdf

    This implementation wraps the imagehash rust crate: https://docs.rs/imagehash/latest/imagehash/

    Parameters
    ----------
    img:
        Pillow image we want to compute the pHash for

    Returns
    -------
    NDArray[np.uint8]
        A length 8 one-dimensional numpy array representing the pHash of the input image.
    """
    rgb_img = img.convert("RGB")

    return hashcalc._native.compute_phash(np.ascontiguousarray(rgb_img))


compute_p_hash = compute_phash


def compute_hamming_distances(
    x: NDArray[np.uint8], ys: NDArray[np.uint8]
) -> NDArray[np.uint32]:
    """Compute the hamming distance between ``x`` and each row of ``ys``.

    The hamming distance between two bit-vectors is the number of differing bits.
    In this case, we count the number of differing bits between two byte-arrays.

    This function is written in Rust for efficiency
    (see https://github.com/emschwartz/hamming-bitwise-fast/blob/main/src/lib.rs)

    Parameters
    ----------
    x:
        One dimensional NumPy array. Must be 8-bit unsigned integers (each int representing eight bits).
        If you have a binary NumPy-array, then you can use :func:`numpy.packbits`-function before calling this function.

    ys:
        Two-dimensional NumPy array. Must be 8-bit unsigned integers (each int representing eight bits).
        The first dimension represents samples and the second dimension must have the same size as the length of ``x``.

    Returns
    -------
    np.ndarray
        One-dimensional NumPy array of same length as the first dimension of ``ys``. Consists of numbers between ``0``
        and ``8 * len(x)``.
    """
    if x.ndim != 1:
        raise ValueError(f"x must be one dimensional, got {x.ndim=}")
    if ys.ndim != 2:
        raise ValueError(f"ys must be two dimensional, got {ys.ndim=}")
    if x.shape[0] != ys.shape[1]:
        raise ValueError(
            f"The hashes have different lengths, {x.shape[0]=} not equal to {ys.shape[1]=}"
        )

    return hashcalc._native.compute_bitwise_hamming_distances(
        x.astype(
            dtype=np.uint8,
            order="C",
            casting="safe",
        ),
        ys.astype(
            dtype=np.uint8,
            order="C",
            casting="safe",
        ),
    )


def compute_hamming_distance(x: NDArray[np.uint8], y: NDArray[np.uint8]) -> int:
    """Compute the hamming distance between byte-vectors ``x`` and ``y``.

    The hamming distance between two bit-vectors is the number of differing bits.
    In this case, we count the number of differing bits between two byte-arrays.

    This function is written in Rust for efficiency
    (see https://github.com/emschwartz/hamming-bitwise-fast/blob/main/src/lib.rs)

    Parameters
    ----------
    x:
        One dimensional NumPy array. Must be 8-bit unsigned integers (each int representing eight bits).
        If you have a binary NumPy-array, then you can use :func:`numpy.packbits`-function before calling this function.

    y:
        One dimensional NumPy array. Must be 8-bit unsigned integers (each int representing eight bits).
        Must have the same length as ``x``.

    Returns
    -------
    int
        The Hamming distance between ``x`` and ``y``. Number between ``0`` and ``8 * len(x)``.
    """
    if x.ndim != 1:
        raise ValueError(f"x must be one dimensional, got {x.ndim=}")
    if y.ndim == 2:
        raise ValueError(
            f"y must be one dimensional, if you have multiple y, use compute_hamming_distances"
        )
    elif y.ndim != 1:
        raise ValueError(f"y must be one dimensional, got {y.ndim=}")

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"The hashes have different lengths, {x.shape[0]=} not equal to {y.shape[0]=}"
        )

    return int(compute_hamming_distances(x, y.reshape((1, -1))).item())
