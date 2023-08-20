from xi.utils import check_args_overlap
from xi.exceptions import XiError
import pytest


def test_overlap_args():
    m = {'1': 1, '2': 2}
    obs = {'4': 2, '3': 1}
    discrete = ['7', '8']

    assert check_args_overlap(m, obs, discrete) == 1


def test_overlap_args_fail_discrete():
    m = {'1': 1, '2': 2}
    obs = {'4': 2, '3': 1}
    discrete = ['3', '8']

    with pytest.raises(XiError) as error:
        check_args_overlap(m, obs, discrete)

    print(f"\n{error.value}")


def test_overlap_args_fail_m():
    m = {'1': 1, '2': 2}
    obs = {'1': 2, '2': 1}
    discrete = ['8']

    with pytest.raises(XiError) as error:
        check_args_overlap(m, obs, discrete)

    print(f"\n{error.value}")
