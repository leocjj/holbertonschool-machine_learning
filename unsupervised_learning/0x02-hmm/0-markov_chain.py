#!/usr/bin/env python3
""" 0x02. Hidden Markov Models """
import numpy as np


def markov_chain(P, s, t=1):
    """
    determines the probability of a markov chain being in a particular
    state after a specified number of iterations
    """
    for _ in range(t):
        s = s @ P
    return s
