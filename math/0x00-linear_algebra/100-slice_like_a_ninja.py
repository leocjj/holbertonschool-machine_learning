#!/usr/bin/env python3
""" 0x00-linear_algebra Task 15 """


def np_slice(matrix, axes={}):
    """  that slices a matrix along specific axes. """

    slicer = [slice(None)] * len(matrix.shape)
    for key, value in axes.items():
        slicer[key] = slice(*value)

    return matrix[tuple(slicer)]
