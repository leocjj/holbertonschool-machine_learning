#!/usr/bin/env python3
""" 0x00-linear_algebra Task 8 """


def dot_prod(arr1, arr2):
    """ performs dod product. """
    if len(arr1) == len(arr2):
        return sum([arr1[i] * arr2[i] for i in range(len(arr1))])


def mat_mul(mat1, mat2):
    """ performs matrix multiplication. """
    if len(mat1[0]) == len(mat2):
        return [[dot_prod(mat1[i], [mat2[k][j] for k in range(len(mat2))])
                 for j in range(len(mat2[0]))]
                for i in range(len(mat1))]
