#!/usr/bin/env python3
""" 0x00-linear_algebra Task 3 """


def matrix_transpose(matrix):
    """ returns the transpose of a matrix. """
    transpose = []
    for col in range(len(matrix[0])):
        temp_row = []
        for row in matrix:
            temp_row.append(row[col])
        transpose.append(temp_row)
    return transpose
