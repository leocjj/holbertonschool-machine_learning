#!/usr/bin/env python3
"""function get the shape of a matrix"""


def matrix_shape(matrix):
    """function get the shape of a matrix"""
    mat_shape = []

    while type(matrix) == list:
        mat_shape.append(len(matrix))
        if type(matrix[0]) == list:
            matrix = matrix[0]
        else:
            break
    return mat_shape