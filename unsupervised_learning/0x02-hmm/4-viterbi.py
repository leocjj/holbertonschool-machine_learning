#!/usr/bin/env python3
""" 0x02. Hidden Markov Models """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
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
    M = Transition.shape[1]

    omega = np.zeros((T, M))
    omega[0, :, np.newaxis] = (Emission[:, Observation[0]] * Initial.T).T

    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            prob = np.multiply(omega[t - 1], Transition[:, j])
            omega[t, j] = np.max(prob) * Emission[j, Observation[t]]
            prev[t - 1, j] = np.argmax(prob)

    S = np.zeros(T)

    last_state = np.argmax(omega[T - 1, :])

    S[-1] = last_state

    back_indx = 1
    for i in range(T - 2, -1, -1):
        S[i] = int(prev[i, int(last_state)])
        last_state = prev[i, int(last_state)]

    return S.astype("int32").tolist(), np.max(omega[T - 1, :])
