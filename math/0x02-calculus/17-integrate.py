#!/usr/bin/env python3
""" 0x02. Calculus Task 17 """


def poly_integral(poly, C=0):
    """  that calculates the integral of a polynomial """
    if not poly or not isinstance(poly, list) or not isinstance(C, int):
        return None
    if len(poly) <= 1:
        return [C]
    result = [C]
    for power, coefficient in enumerate(poly):
        temp = coefficient / (power + 1)
        if temp.is_integer():
            temp = int(temp)
        result.append(temp)
    return result
