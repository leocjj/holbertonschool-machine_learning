#!/usr/bin/env python3
""" 0x02. Hidden Markov Models """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray)\
            or len(Transition.shape) != 2\
            or Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Emission.shape[0] != Transition.shape[0] != Transition.shape[0] !=\
       Initial.shape[0]:
        return None, None
    if Initial.shape[1] != 1:
        return None, None

    T = Observation.shape[0]
    N = Transition.shape[0]

    alpha = np.zeros((N, T))
    alpha[:, 0, np.newaxis] = (Initial.T * Emission[:, Observation[0]]).T

    for t in range(1, Observation.shape[0]):
        for j in range(Transition.shape[0]):
            alpha[j, t] = (alpha[:, t - 1].dot(Transition[:, j]) *
                           Emission[j, Observation[t]])

    prob = np.sum(alpha[:, -1])

    return prob, alpha
