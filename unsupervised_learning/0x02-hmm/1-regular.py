#!/usr/bin/env python3
""" 0x02. Hidden Markov Models """
import numpy as np


def regular(P):
    """ determines if a markov chain is absorbing """
    try:
        dim = P.shape[0]
        q = (P - np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q, ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        answer = np.linalg.solve(QTQ, bQT)
        if np.all(answer > 0):
            return answer
        else:
            return None
    except Exception as e:
        return None
