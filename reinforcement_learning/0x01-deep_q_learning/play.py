#!/usr/bin/env python3
"""
play.py
Script that can display a game played by the agent trained by train.py:
* Load the policy network saved in policy.h5
* Your agent should use the GreedyQPolicy
"""
import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
import tensorflow.keras as K
AtariProcessor = __import__('train').AtariProcessor
create_q_model = __import__('train').create_q_model


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n

    # screenshots per state
    window = 4
    # Deep Q-Network
    model = create_q_model(num_actions, window)
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()

    dqn = DQNAgent(model=model, nb_actions=num_actions,
                   policy=GreedyQPolicy(),
                   processor=processor, memory=memory)

    dqn.compile(K.optimizers.Adam(learning_rate=.00025), metrics=['mae'])

    # Load the policy network
    dqn.load_weights('policy.h5')

    # Only works with 'visualize=False' if in Colab
    dqn.test(env, nb_episodes=10, visualize=True)
