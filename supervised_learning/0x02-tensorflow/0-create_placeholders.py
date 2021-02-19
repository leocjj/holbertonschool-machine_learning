#!/usr/bin/env python3
""" 0x01. Classification """
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Function that returns two placeholders, x and y, for the neural network
    x is the placeholder for the input data to the neural network
    y is the placeholder for the one-hot labels for the input data
    :param nx: the number of feature columns in our data
    :param classes: the number of classes in our classifier
    :return: placeholders named x and y, respectively
    """

    x = tf.placeholder("float", shape=[None, nx], name='x')
    y = tf.placeholder("float", shape=[None, classes], name='y')

    return x, y