#!/usr/bin/env python3
""" 0x02. Calculus Task 9 """


def summation_i_squared(n):
    """ that calculates sum of i power of 2 from 1 to n """
    if not isinstance(n, int):
        return None
    return int(sum([i**2 for i in range(1, n + 1)]))
