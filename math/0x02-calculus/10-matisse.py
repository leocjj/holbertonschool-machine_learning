#!/usr/bin/env python3
""" 0x02. Calculus Task 10 """


def poly_derivative(poly):
    """  that calculates the derivative of a polynomial """
    if not poly or not isinstance(poly, list):
        return None
    if len(poly) <= 1:
        return [0]
    return [coefficient * power for power, coefficient in enumerate(poly)
            if power >= 1]
