#!/usr/bin/env python3
""" 0x02. Tensorflow """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuracy of a prediction.
    :param y: is a placeholder for the labels of the input data
    :param y_pred: is a tensor containing the network’s predictions
    :return: a tensor containing the decimal accuracy of the prediction
    """

    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy
