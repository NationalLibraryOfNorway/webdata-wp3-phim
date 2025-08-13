import pytest
from hashcalc.hashcalc import hamming_distance


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
def test_known_examples(num1, num2, distance):
    assert hamming_distance(num1, num2) == distance
