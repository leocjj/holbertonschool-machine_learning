#!/usr/bin/env python3
"""function get the shape of a matrix"""


def matrix_shape(matrix):
    """ calculates the shape of a matrix """
    shape = []
    if not matrix:
        return 0
    if isinstance(matrix[0], int):
        return len(matrix)
    if isinstance(matrix[0], list):
        shape.append(len(matrix))
        len1 = matrix_shape(matrix[0])
        if isinstance(len1, int):
            shape.append(len1)
        if isinstance(len1, list):
            shape.extend(len1)
        return shape
