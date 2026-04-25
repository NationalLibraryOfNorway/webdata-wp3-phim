"""Utilities for comparing images with perceptual hashes.

All functions wrap implementations in low-level languages (Rust for pHash and hamming distances, C++ for PDQ-hash).
"""

from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

import phim._native

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
    # Import here since pdqhash is an optional dependency
    import pdqhash  # noqa: I

    hash_vectors, quality = pdqhash.compute_dihedral(np.array(img))
    return np.packbits(hash_vectors, axis=1), quality


def compute_phash(
    image: Image.Image,
    hash_size: int = 8,
    dct_size: int = 32,
    antialias: int = Image.Resampling.LANCZOS,
    threshold_fn: Callable[[np.ndarray], float] = np.median,
    handle_all_black_and_transparent_images: bool = True,
) -> np.typing.NDArray[np.uint8]:
    """Compute the pHash (DCT) of an image.

    The pHash is based on the discrete cosine transform (DCT) of the image. In particular, the image is resized to
    32 x 32 pixels by default (hash_size x hash_size).
    Then, we compute the DCT of the image and get the 64 lowest frequencies omitting the lowest ones (8-by-8 pixel
    square shifted one row and column from the top left by default. hash_size x hash_size in general).
    Finally, we find the average value of these frequencies, and threshold the DCT matrix based on it to get a binary
    pattern, which constitutes our 64-bit hash.

    For more information, see Chapter 3.2.1 of Christoph Zauner's master's thesis:
    https://www.phash.org/docs/pubs/thesis_zauner.pdf

    Parameters
    ----------
    img:
        Pillow image we want to compute the pHash for
    hash_size:
        Number of bytes in the hash. 8 by default.
    dct_size:
        The size that the image is resized to before computing the dct. 32 by default.
        Note that the `dct_size` needs to be at least hash_size+1`.
    antialias:
        Antialias method used when resizing. `Image.Resampling.LANCZOS` by default.
    threshold_fn:
        Function used to find threshold for binarizing the dct.
    handle_all_black_and_transparent_images:
        If True, then the hash for images with only black and transparent pixels will be computed by alpha-compositing
        the image onto a white background. This is neccessary as Pillow by default composites the alpha channel over
        black when it's removed. For black images, this leads to a completely black image, and an all-zero hash.
        Therefore, if ``image`` has transparency and the a black backround leads to a zero-valued hash, then the image
        will be alpha-composited onto a white background instead.

    Returns
    -------
    NDArray[np.uint8]
        A length `hash_size^2/8` one-dimensional numpy array representing the pHash of the input image.
    """
    # Import here since scipy is an optional dependency
    import scipy  # noqa: I

    if dct_size < hash_size + 1:
        raise ValueError(f"`dct_size` needs to be at least `hash_size+1`, got `{dct_size =}` and `{hash_size+1 =}`")

    image = image.resize((dct_size, dct_size), antialias)
    greyscale_image = image.convert("L")
    dct = scipy.fft.dctn(np.asarray(greyscale_image), type=2, axes=(0, 1))[1 : hash_size + 1, 1 : hash_size + 1]

    # For transparent images: alpha-composite over white instead of black for all-zero hashes (see docstring).
    if image.has_transparency_data and handle_all_black_and_transparent_images and not dct.any():
        white_background = Image.new("LA", (dct_size, dct_size), color="white")
        white_background.alpha_composite(image.convert("LA"))
        white_background.convert("L")
        dct = scipy.fft.dctn(np.asarray(white_background), type=2, axes=(0, 1))[1 : hash_size + 1, 1 : hash_size + 1]

    return np.packbits(dct > threshold_fn(dct))


compute_p_hash = compute_phash


def compute_hamming_distances(x: NDArray[np.uint8], ys: NDArray[np.uint8]) -> NDArray[np.uint32]:
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
        raise ValueError(f"The hashes have different lengths, {x.shape[0]=} not equal to {ys.shape[1]=}")

    return phim._native.compute_bitwise_hamming_distances(
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
        raise ValueError(f"y must be one dimensional, if you have multiple y, use compute_hamming_distances")
    elif y.ndim != 1:
        raise ValueError(f"y must be one dimensional, got {y.ndim=}")

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"The hashes have different lengths, {x.shape[0]=} not equal to {y.shape[0]=}")

    return int(compute_hamming_distances(x, y.reshape((1, -1))).item())
