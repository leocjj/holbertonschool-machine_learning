#!/usr/bin/env python3
"""
0x00. Q-learning
"""
import numpy as np
load_frozen_lake = __import__('0-load_env').load_frozen_lake


def epsilon_greedy(Q, state, epsilon):
    env = load_frozen_lake()
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action