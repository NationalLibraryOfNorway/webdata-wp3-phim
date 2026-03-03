import numpy as np
import pytest
from hashcalc import compute_hamming_distance
from numpy.typing import NDArray


@pytest.mark.parametrize(
    "num1, num2, distance",
    [
        (0b0000, 0b0000, 0),
        (0b0001, 0b0001, 0),
        (0b1101, 0b1011, 2),
        (0b1111, 0b0000, 4),
        (0b101010, 0b010101, 6),
    ],
)
def test_known_examples(num1: int, num2: int, distance: int) -> None:
    x = np.array([num1], dtype=np.uint8)
    y = np.array([num2], dtype=np.uint8)
    assert compute_hamming_distance(x, y) == distance


@pytest.mark.parametrize(
    "x",
    [
        np.array([[0], [1]], dtype=np.uint8),
        np.array(1, dtype=np.uint8),
        np.array([[1, 2, 3, 4], [1, 2, 1, 2]], dtype=np.uint8),
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
    ],
)
def test_x_has_dim_1(x: NDArray[np.uint8]) -> None:
    y = np.array([1, 2, 3, 4], dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_hamming_distance(x, y)


@pytest.mark.parametrize(
    "y",
    [
        np.array([[0], [1]], dtype=np.uint8),
        np.array(1, dtype=np.uint8),
        np.array([[1, 2, 3, 4], [1, 2, 1, 2]], dtype=np.uint8),
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
    ],
)
def test_y_has_dim_1(y: NDArray[np.uint8]) -> None:
    x = np.array([1, 2, 3, 4], dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_hamming_distance(x, y)
