#!/usr/bin/env python3
""" 0x02. Calculus Task 10 """


def poly_derivative(poly):
    """  that calculates the derivative of a polynomial """
    result = []
    if not poly:
        return None
    for power, coefficient in enumerate(poly):
        if power >= 1:
            result.append(coefficient * power)
    return result
