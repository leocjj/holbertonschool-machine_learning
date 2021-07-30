#!/usr/bin/env python3
""" 0x07. Bayesian Probability """
from scipy import special


def posterior(x, n, p1, p2):
    """
    Returns: the posterior probability that
    p is within the range [p1, p2] given x and n
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")

    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    """
    Cumulative distribution function of the beta distribution.
    Returns the integral from zero to u
    of the beta probability density function.
    btdtr(a, b, u)
    a: Shape parameter (a > 0)
    b: Shape parameter (b > 0)
    u: Upper limit of integration, in [0, 1]
    """
    # x follows a binomial distribution
    # Relation between beta distribution and binomial distribution
    a = x + 1
    b = n - x + 1

    cdf_beta1 = special.btdtr(a, b, p1)
    cdf_beta2 = special.btdtr(a, b, p2)

    Posterior = cdf_beta2 - cdf_beta1

    return Posterior
