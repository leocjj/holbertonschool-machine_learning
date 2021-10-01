#!/usr/bin/env python3
"""
0x12 Transformer applications
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation:
    inputs is a tf.Tensor of shape (batch_size, seq_len_in) that contains the
        input sentence
    target is a tf.Tensor of shape (batch_size, seq_len_out) that contains the
        target sentence
    This function should only use tensorflow operations in order to properly
        function in the training step
    Returns: encoder_mask, combined_mask, decoder_mask
    encoder_mask is the tf.Tensor padding mask of shape
        (batch_size, 1, 1, seq_len_in) to be applied in the encoder
    combined_mask is the tf.Tensor of shape
        (batch_size, 1, seq_len_out, seq_len_out) used in the 1st attention
        block in the decoder to pad and mask future tokens in the input
        received by the decoder. It takes the maximum between a look ahead mask
        and the decoder target padding mask.
    decoder_mask is the tf.Tensor padding mask of shape
        (batch_size, 1, 1, seq_len_in) used in the 2nd attention block in the
        decoder.
    """

    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    size = tf.shape(target)[1]
    ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    dec_target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_mask = dec_target_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(dec_target_mask, ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
