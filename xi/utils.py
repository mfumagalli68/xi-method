import itertools

from exceptions import *

def partition_validation(arg, k):

    if isinstance(arg,float):
        raise TypeError('Argument need to be a positive integer')

    if isinstance(arg,int):
        if arg <= 0:
            raise ValueError('Argument could only be a strictly positive integer'
                             'or a dictionary.')
    if isinstance(arg,dict):
        keys = len(list(arg.keys()))
        if keys>k:
            raise XiError("Number of keys of dictionary specifying partions"
                          "is greater than number of features")


def check_args_overlap(*args):

    overlap = []
    sets = tuple(set(d.keys()) for d in args)
    prods = itertools.combinations(sets,r=2)
    for s in prods:
        overlap.extend(set.intersection(*s))

    overlap = set(overlap)
    if len(overlap)>0:
        raise XiError(f"m,obs,discrete parameters can\'t have the same keys.\n"
                      f"Overlapping keys {overlap}.\n"
                      f"Please specify dictionaries differently.")




if __name__=='__main__':
    m = {'1':1,'2':2}
    obs = {'1': 1, '4': 2, '3': 1}
    discrete = {'1': 1, '2': 2}

    check_args_overlap(m,obs,discrete)
