#!/usr/bin/env python3
""" 0x02. Hidden Markov Models """
import numpy as np


def markov_chain(P, s, t=1):
    """ determines the steady state probabilities of a regular markov chain """
    for _ in range(t):
        s = s @ P
    return s
