#!/usr/bin/env python3
""" 0x00-linear_algebra Task 16 """
from numpy import array


def add_matrices(mat1, mat2):
    """  that adds two matrices. """
    m1 = array(mat1)
    m2 = array(mat2)
    try:
        result = m1 + m2
    except ValueError:
        return None
    return result
