import itertools
import logging
import time
from functools import wraps
from typing import *

from xi_method.exceptions import XiError
from xi_method.separation.measurement import builder_mapping


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Total execution time: {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def partition_validation(arg: Union[int, Dict, float,List], k: int) -> None:
    """
    Validate partition specified by the user

    :param arg:
    :param k:
    :return:
    """
    if isinstance(arg, float):
        raise TypeError(f"Number of partitions need to be a positive integer")

    if isinstance(arg, int):
        if arg <= 0:
            raise ValueError(f"Number of partitions could only be a strictly positive integer"
                             f"or a dictionary.")
    if isinstance(arg, dict):
        keys = len(list(arg.keys()))
        if keys > k:
            raise XiError("Number of keys of dictionary specifying partitions"
                          "is greater than number of features")


def separation_measurement_validation(measure: Union[List, AnyStr]) -> Union[XiError,int]:
    """
    Validate separation measurement, if not implemented throw an error

    :param measure:
    :return:
    """
    if isinstance(measure, str):
        measure = [measure]
    for m in measure:
        if m not in builder_mapping.keys():
            raise XiError(f"Separation measurement {m} not implemented. Please choose "
                          f"one or more than one from {list(builder_mapping.keys())}")

    return 1

def get_separation_measurement() -> AnyStr:
    """
    Validate separation measurement, if not implemented throw an error

    :param measure:
    :return:
    """

    seps = '\n'.join(list(builder_mapping.keys()))
    logging.info(f"These are the separation measurement implemented in this package: " \
           f"{seps}")

def check_args_overlap(*args):
    """

    :param args:
    :return:
    """
    if any(isinstance(i, int) for i in args):
        return 0
    overlap = []
    sets = tuple(set(d) for d in args)
    prods = itertools.combinations(sets, r=2)
    for s in prods:
        overlap.extend(set.intersection(*s))

    overlap = set(overlap)
    if len(overlap) > 0:
        raise XiError(f"Parameter m, obs, discrete can\'t have the same keys.\n"
                      f"Overlapping keys {overlap}.\n"
                      f"Please specify parameters differently.")
    else:
        return 1
