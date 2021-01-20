#!/usr/bin/env python3
""" 0x00-linear_algebra Task 15 """
import numpy as np


def add_matrices(mat1, mat2):
    """  that adds two matrices:. """
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    try:
        result = m1 + m2
    except Exception:
        return None
    return result
