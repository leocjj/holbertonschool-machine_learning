#!/usr/bin/env python3
""" 0x00-linear_algebra Task 5 """


def add_matrices2D(mat1, mat2):
    """ adds two matrices element-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j]
             for j, _ in enumerate(mat1[0])]
            for i, _ in enumerate(mat1)]
