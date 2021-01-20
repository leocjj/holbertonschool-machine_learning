#!/usr/bin/env python3
""" 0x00-linear_algebra Task 6 """


def cat_arrays(arr1, arr2):
    """ concatenates two arrays. """
    len12 = len(arr1) + len(arr2)
    return [*arr1, *arr2]
