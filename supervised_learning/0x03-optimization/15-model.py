#!/usr/bin/env python3
""" 0x03. Optimization """

import tensorflow as tf
import numpy as np


def create_layer(prev, n, activation):
    """
    Function to create a layer
    :param prev: is the tensor output of the previous layer
    :param n: is the number of nodes in the layer to create
    :param activation: is the activation function that the layer should use
    :return: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, name='layer',
                            kernel_initializer=init)
    return layer(prev)


def forward_prop(x, layer, activations):
    """
    Function that creates the forward propagation graph for the neural network.
    :param x: placeholder for input data
    :param layer: list containing the number of nodes in each layer.
    :param activations: list containing activation function for each layer.
    :return: the prediction of the network in tensor form
    """

    # First layer
    y_pred = create_batch_norm_layer(x, layer[0], activations[0])

    for i in range(1, len(layer)):
        y_pred = create_batch_norm_layer(y_pred, layer[i], activations[i])

    return y_pred


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuracy of a prediction.
    :param y: is a placeholder for the labels of the input data
    :param y_pred: is a tensor containing the network’s predictions
    :return: a tensor containing the decimal accuracy of the prediction
    """

    prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy


def calculate_loss(y, y_pred):
    """
    Function that calculates the softmax cross-entropy loss of a prediction.
    :param y: is a placeholder for the labels of the input data.
    :param y_pred: is a tensor containing the network’s predictions.
    :return: a tensor containing the loss of the prediction.
    """

    return tf.losses.softmax_cross_entropy(y, y_pred)


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way:
    :param X: is the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features
    :param Y: is the second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y
    :return: shuffled X and Y matrices
    """

    shuffler = np.random.permutation(len(X))

    return X[shuffler], Y[shuffler]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
        the Adam optimization algorithm:
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta1: weight used for the first moment
    :param beta2: weight used for the second moment
    :param epsilon: a small number to avoid division by zero
    :return: Adam optimization operation
    """

    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    adam = optimizer.minimize(loss)

    return adam


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow:
        Your layer should incorporate two trainable parameters, gamma and beta,
        initialized as vectors of 1 and 0 respectively. You should use an
        epsilon of 1e-8
    :param prev: activated output of the previous layer
    :param n: number of nodes in the layer to be created
    :param activation: activation function that should be used on the output
        of the layer
    :return: tensor of the activated output for the layer
    """

    if activation is None:
        A = create_layer(prev, n, activation)
        return A

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    y = tf.layers.Dense(units=n, kernel_initializer=init, name='layer')
    x = y(prev)

    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    epsilon = 1e-8

    norma = tf.nn.batch_normalization(x=x, mean=mean, variance=variance,
                                      offset=beta, scale=gamma,
                                      variance_epsilon=epsilon, name='Z')
    return activation(norma)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time
    decay. The learning rate decay should occur in a stepwise fashion.
    :param alpha: original learning rate
    :param decay_rate: used to determine the rate at which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further
    :return: learning rate decay operation
    """

    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using Adam
    optimization, mini-batch gradient descent, learning rate decay, and batch
    normalization:
    :param Data_train:
    :param Data_valid:
    :param layers:
    :param activations:
    :param alpha:
    :param beta1:
    :param beta2:
    :param epsilon:
    :param decay_rate:
    :param batch_size:
    :param epochs:
    :param save_path:
    :return:
    """
    X_train, Y_train = Data_train[0], Data_train[1]
    X_valid, Y_valid = Data_valid[0], Data_valid[1]

    steps = int(np.ceil(X_train.shape[0] / batch_size))

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    tf.add_to_collection('x', x)
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    # Add ops to save/restore variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                # learning rate decay
                sess.run(global_step.assign(epoch))
                # update learning rate
                sess.run(alpha)
                # shuffle data, both training set and labels
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                # mini-batch within epoch
                for step_number in range(steps):
                    # data selection mini batch from training set and labels
                    start = step_number * batch_size
                    end = (step_number + 1) * batch_size
                    if end > X_train.shape[0]:
                        end = X_train.shape[0]
                    X = X_shuffled[start:end]
                    Y = Y_shuffled[start:end]
                    # execute training for step
                    sess.run(train_op, feed_dict={x: X, y: Y})
                    if step_number != 0 and (step_number + 1) % 100 == 0:
                        # step_number is the number of times gradient
                        # descent has been run in the current epoch
                        print("\tStep {}:".format(step_number + 1))
                        # calculate cost and accuracy for step
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X, y: Y})
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
