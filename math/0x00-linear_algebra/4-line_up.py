#!/usr/bin/env python3
""" 0x00-linear_algebra Task 4 """


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
