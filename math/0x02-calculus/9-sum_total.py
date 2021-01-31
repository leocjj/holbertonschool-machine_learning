#!/usr/bin/env python3
""" 0x02. Calculus Task 9 """


def summation_i_squared(n):
    """ that calculates sum of i power of 2 from 1 to n """
    if not isinstance(n, int) or n < 1:
        return None
    return sum(map(lambda x: x**2, range(1, n + 1)))
