#!/usr/bin/env python3
""" 0x00-linear_algebra Task 13 """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """  concatenates two matrices along a specific axis. """
    return np.concatenate((mat1, mat2), axis)
