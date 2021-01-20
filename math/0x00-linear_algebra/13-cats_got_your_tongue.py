#!/usr/bin/env python3
""" 0x00-linear_algebra Task 13 """
from numpy import concatenate


def np_cat(mat1, mat2, axis=0):
    """  concatenates two matrices along a specific axis. """
    return concatenate((mat1, mat2), axis)
