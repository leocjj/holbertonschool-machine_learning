#!/usr/bin/env python3
"""
train.py
Script that utilizes keras, keras-rl, and gym
to train an agent that can play Atari’s Breakout:
* Use keras-rl‘s DQNAgent, SequentialMemory, and EpsGreedyQPolicy
* Save the final policy network as policy.h5
"""
import gym
import numpy as np
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor
import tensorflow.keras as K
from tensorflow.keras import layers


class AtariProcessor(Processor):
    """
    The environment in which the game will be played.
    Processor for Atari.
    Prepocesses data based on Deep Learning
    Quick Reference by Mike Bernico.
    """
    def process_observation(self, observation):
        """
        Resizing and grayscale
        """
        # (height, width, channel)
        assert observation.ndim == 3
        # Retrieve image from array
        img = Image.fromarray(observation)
        # Resize image and convert to grayscale
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')
        # Convert back to array
        processed_observation = np.array(img)
        # Assert input shape
        assert processed_observation.shape == (84, 84)

        # Save processed observation in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Convert the batch of images to float32
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """
        Rewards between -1 and 1
        """
        return np.clip(reward, -1., 1.)


def create_q_model(num_actions, window):
    """
    CNN with Keras defined by the Deepmind paper
    """
    # Each RL state is composed of 4 windows
    inputs = layers.Input(shape=(window, 84, 84))
    # Permute is used to change the dimensions of the input
    # according to a given pattern
    layer0 = layers.Permute((2, 3, 1))(inputs)

    layer1 = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4),
                           activation="relu",
                           data_format="channels_last")(layer0)
    layer2 = layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2),
                           activation="relu",
                           data_format="channels_last")(layer1)
    layer3 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1),
                           activation="relu",
                           data_format="channels_last")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return K.Model(inputs=inputs, outputs=action)


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n
    window = 4
    model = create_q_model(num_actions, window)
    model.summary()
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy,
                   memory=memory, processor=processor,
                   nb_steps_warmup=50000, gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)
    dqn.compile(K.optimizers.Adam(learning_rate=.00025), metrics=['mae'])
    dqn.fit(env,
            nb_steps=10000000,
            log_interval=10000,
            visualize=False,
            verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
