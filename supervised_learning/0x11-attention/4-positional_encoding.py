#!/usr/bin/env python3
"""
0x11. Attention
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer:
    max_seq_len is an integer representing the maximum sequence length
    dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    d_model = dm
    rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    rads = pos * rates

    rads[:, 0::2] = np.sin(rads[:, 0::2])
    rads[:, 1::2] = np.cos(rads[:, 1::2])

    return rads
