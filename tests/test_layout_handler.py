import pytest

from crossword_generator.common import Orientation
from crossword_generator.layout_handler import get_slice, get_coordinates


@pytest.mark.parametrize(
    "row, col, orientation, expected1, expected2",
    [
        (1, 2, Orientation.ACROSS, slice(1, 1 + 1, None), slice(2, None, None)),
        (5, 6, Orientation.ACROSS, slice(5, 5 + 1, None), slice(6, None, None)),
        (7, 8, Orientation.DOWN, slice(7, None, None), slice(8, 8 + 1, None)),
    ],
)
def test_get_slice(row, col, orientation, expected1, expected2):
    slice1, slice2 = get_slice(row, col, orientation)

    assert slice1 == expected1
    assert slice2 == expected2


@pytest.mark.parametrize(
    "row, col, orientation, length, expected",
    [
        (5, 6, Orientation.ACROSS, 2, [(5, 6), (5, 7)]),
        (7, 8, Orientation.DOWN, 3, [(7, 8), (8, 8), (9, 8)]),
    ],
)
def test_get_coordinates(row, col, orientation, length, expected):
    result = get_coordinates(row, col, orientation, length)

    assert result == expected
