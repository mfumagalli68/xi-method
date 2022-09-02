from exceptions import *

def m_validation(arg,k):

    if isinstance(arg,float):
        raise TypeError('m argument need to be a positive integer')

    if isinstance(arg,int):
        if arg <= 0:
            raise ValueError('m argument could only be a strictly positive integer'
                             'or a dictionary.')
    if isinstance(arg,dict):
        keys = len(list(arg.keys()))
        if keys>k:
            raise XiError("Number of keys of dictionary specifying partions"
                          "is greater than number of features")


