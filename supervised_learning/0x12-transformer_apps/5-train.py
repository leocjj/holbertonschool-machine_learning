#!/usr/bin/env python3
"""
0x12 Transformer applications
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """ """
    pass
