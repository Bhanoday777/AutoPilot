import pytest

from calc.stats import mean, median


def test_mean() -> None:
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_empty_list() -> None:
    with pytest.raises(ValueError):
        mean([])


def test_median_odd_length() -> None:
    assert median([3, 1, 2]) == 2
