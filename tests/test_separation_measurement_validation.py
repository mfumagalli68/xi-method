import pytest
from xi.ximp import *
from xi.utils import separation_measurement_validation


def test_separation_measurement_validation_fail():
    measure = 'unknown'
    with pytest.raises(XiError) as error:
        separation_measurement_validation(measure)

    print(f"\n{error.value}")


def test_separation_measurement_validation_mix():
    measure = ['Kuiper', 'Unknown']
    with pytest.raises(XiError) as error:
        separation_measurement_validation(measure)

    print(f"\n{error.value}")


def test_separation_measurement_validation():
    measure = ['Kuiper', 'Hellinger']

    assert separation_measurement_validation(measure) == 1
