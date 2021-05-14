#!/usr/bin/env python3
""" 0x02. Tensorflow """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Function that evaluates the output of a neural network.
    :param X: is a numpy.ndarray containing the input data to evaluate.
    :param Y: is a numpy.ndarray containing the one-hot labels for X.
    :param save_path: is the location to load the model from.
    :return: network’s prediction, accuracy, and loss, respectively.
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        test_prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        test_accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        test_cost = sess.run(loss, feed_dict={x: X, y: Y})

    return test_prediction, test_accuracy, test_cost
