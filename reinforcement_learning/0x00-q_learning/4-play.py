#!/usr/bin/env python3
"""
0x00. Q-learning
"""
import numpy as np
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the frozen lake"""
    env = gym.make("FrozenLake-v0", desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env


def epsilon_greedy(Q, state, epsilon):
    """always exploit q table"""
    env = load_frozen_lake()
    # print(env)
    action = np.argmax(Q[state, :])
    return action


def train(env,
          Q,
          episodes=1,
          max_steps=100,
          alpha=0.1,
          gamma=0.99,
          epsilon=1,
          min_epsilon=0.1,
          epsilon_decay=0.05):
    """
    performs Q-learning:
    env: is the FrozenLakeEnv instance
    Q: is a numpy.ndarray containing the Q-table
    episodes: is the total number of episodes to train over
    max_steps: is the maximum number of steps per episode
    alpha: is the learning rate
    gamma: is the discount rate
    epsilon: is the initial threshold for epsilon greedy
    min_epsilon: is the minimum value that epsilon should decay to
    epsilon_decay: is the decay rate for updating epsilon between episodes
    Returns: Q, total_rewards
        Q: is the updated Q-table
        total_rewards: is a list containing the rewards per episode
    """

    rewards = []
    for _ in range(episodes):
        state = env.reset()
        if epsilon < min_epsilon:
            epsilon = min_epsilon
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            Q[state][action] += epsilon_decay * (reward + gamma *
                                                 np.max(Q[new_state]) -
                                                 Q[state][action])
            state = new_state
            env.render()
            if done:
                break
        rewards.append(reward)

    return Q, rewards


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode:
    env: is the FrozenLakeEnv instance
    Q: is a numpy.ndarray containing the Q-table
    max_steps: is the maximum number of steps in the episode
    Each state of the board should be displayed via the console
    You should always exploit the Q-table
    Returns: the total rewards for the episode
    """
    Q, rewards = train(env, Q, max_steps=max_steps)

    return np.max(rewards)