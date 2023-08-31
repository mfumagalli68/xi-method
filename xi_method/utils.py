import itertools
import logging
import time
from functools import wraps
from typing import *
import pandas as pd

from xi_method.exceptions import XiError
from xi_method.separation.measurement import builder_mapping
from xi_method import _ROOT

def load_wine_quality_red_dataset():

    path = _ROOT / 'data' / 'winequality-red.csv'
    data = pd.read_csv(path,sep=";")
    return data

def load_bottle_dataset():

    path = _ROOT / 'data' / 'bottle.csv'
    data = pd.read_csv(path,sep=",")
    return data

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


def partition_validation(arg: Union[int, Dict, float, List], k: int) -> None:
    """
    Validate partition specified by the user

    :param arg: m,obs,discrete

    :param k: number of covariates

    :return: Throw an exception if partitions doesn't respect criteria ( negative, float or number of partitions specification
    greater than number of covariates)
    """
    if isinstance(arg, float):
        raise TypeError(f"Number of partitions need to be a positive integer")

    if isinstance(arg, int):
        if arg <= 0:
            raise ValueError(f"Number of partitions could only be a strictly positive integer or a dictionary.")
    if isinstance(arg, dict):
        keys = len(list(arg.keys()))
        if keys > k:
            raise XiError("Number of keys of dictionary specifying partitions"
                          "is greater than number of features")


def separation_measurement_validation(measure: Union[List, AnyStr]) -> Union[XiError, int]:
    """
    Validate separation measurement choice by the user, if not implemented throw an error

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


def get_separation_measurement() -> None:
    """
    Get a list of implemented separation measurement

    :return: None, prnt a list of implemented separation measurement
    """

    seps = '\n'.join(list(builder_mapping.keys()))
    logging.info(f"These are the separation measurement implemented in this package: " \
                 f"{seps}")


def check_args_overlap(*args) -> None:

    """
    Check if partition argument overlaps.
    Users can't specify more than one partition for covariate

    :param args: m,obs,discrete : partition parameter
    
    :return: Throw an exception if a key is found more than one in the args.
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

