from typing import Any

import numpy as np
import pytest
from hashcalc import compute_hamming_distances
from numpy.typing import NDArray


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@pytest.mark.parametrize(
    "x, error_type",
    [
        (np.array([1]), ValueError),
        (np.array([[1, 2], [3, 4]]), ValueError),
        (np.array(["a", "b"]), TypeError),
        (np.array([0.1, 0.2]), TypeError),
        (np.array([512, 513]), TypeError),
    ],
)
def test_invalid_datatypes_x(x: Any, error_type: type) -> None:
    """compyte_hamming_dinstances_will give typerror for invalid inputtypes"""
    ys = np.array([[1, 2], [3, 4]])
    with pytest.raises(error_type):
        compute_hamming_distances(x, ys)


@pytest.mark.parametrize(
    "ys, error_type",
    [
        (np.array([1, 2, 3]), ValueError),
        (np.array([[0.1, 0.2], [0.3, 0.4]]), TypeError),
        (np.array([["a", "b"], ["c", "d"]]), TypeError),
        (np.array([[512, 513], [514, 515]]), TypeError),
    ],
)
def test_invalid_datatypes_ys(ys: Any, error_type: type) -> None:
    """compute_hamming_dinstances_will give typerror for invalid inputtypes"""
    x = np.array([1, 2])
    with pytest.raises(error_type):
        compute_hamming_distances(x, ys)


@pytest.mark.parametrize(
    "x",
    [
        np.array([1], dtype=np.uint8),
        np.array([0], dtype=np.uint8),
    ],
)
@pytest.mark.parametrize(
    "ys",
    [
        np.array([[0], [0]], dtype=np.uint8),
        np.array([[1], [3]], dtype=np.uint8),
        np.array([[1], [0]], dtype=np.uint8),
    ],
)
def test_same_result_as_python_implementation_uint8(
    x: NDArray[np.uint8], ys: NDArray[np.uint8]
) -> None:
    distances = compute_hamming_distances(x, ys)
    for distance, y in zip(distances, ys):
        assert distance == hamming_distance(int(x.squeeze()), int(y.squeeze()))


@pytest.mark.parametrize(
    "x",
    [
        np.array([1, 2, 3, 4], dtype=np.uint8),
        np.array([1, 2, 1, 2], dtype=np.uint8),
    ],
)
@pytest.mark.parametrize(
    "ys",
    [
        np.array([[1, 2, 3, 4], [1, 2, 1, 2]], dtype=np.uint8),
        np.array([[0, 0, 0, 0]], dtype=np.uint8),
    ],
)
def test_same_result_as_python_implementation_uint32(
    x: NDArray[np.uint8], ys: NDArray[np.uint8]
) -> None:
    distances = compute_hamming_distances(x, ys)
    for distance, y in zip(distances, ys):
        x_uint32 = x.view(np.uint32)
        y_uint32 = y.view(np.uint32)
        assert distance == hamming_distance(
            int(x_uint32.squeeze()), int(y_uint32.squeeze())
        )
