#!/usr/bin/env python3
"""
0x00. Q-learning
"""
import numpy as np
load_frozen_lake = __import__('0-load_env').load_frozen_lake


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action:
    Q is a numpy.ndarray containing the q-table
    state is the current state
    epsilon is the epsilon to use for the calculation
    You should sample p with numpy.random.uniformn to determine if your
        algorithm should explore or exploit
    If exploring, you should pick the next action with numpy.random.randint
        from all possible actions
    Returns: the next action index
    """
    env = load_frozen_lake()

    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])

    return action