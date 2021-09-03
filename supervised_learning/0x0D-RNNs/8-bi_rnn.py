#!/usr/bin/env python3
"""
Performs forward propagation for a bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Returns: H, Y
    """
    t, m, _ = X.shape
    _, h = h_0.shape
    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))
    Hf[0] = h_0
    Hb[t] = h_t

    for fw_step in range(t):
        Hf[fw_step + 1] = bi_cell.forward(Hf[fw_step], X[fw_step])

    for bw_step in range(t - 1, -1, -1):
        Hb[bw_step] = bi_cell.backward(Hb[bw_step + 1], X[bw_step])

    H = np.concatenate((Hf[1:], Hb[0:t]), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
