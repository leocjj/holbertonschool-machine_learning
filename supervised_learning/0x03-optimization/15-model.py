#!/usr/bin/env python3
""" 0x03. Optimization """
import tensorflow as tf
import numpy as np


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


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that creates the forward propagation graph for the neural network.
    :param x: is the placeholder for input data
    :param layer_sizes: is a list containing the number of nodes in each layer.
    :param activations: a list containing activation function for each layer.
    :return: the prediction of the network in tensor form
    """

    layer = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])

    return layer


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


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy:
    The learning rate decay should occur in a stepwise fashion.
    :param alpha: learning rate
    :param decay_rate: used to determine the rate at which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further
    :return: updated value for alpha
    """

    alpha /= 1 + decay_rate * np.floor(global_step / decay_step)
    return alpha


def Adam_op(loss, alpha, beta1, beta2, epsilon):
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


def shuffle_data(X, Y):
    """
    Normalizes (standardizes) a matrix:
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


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """

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

    # create placeholders
    x, y = create_placeholders(Data_train[0].shape[1], Data_train[1].shape[1])
    # forward propagation
    y_pred = forward_prop(x, layers, activations)
    # accuracy
    accuracy = calculate_accuracy(y, y_pred)
    # loss with softmax entropy
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # learning rate decay
    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign(global_step, global_step + 1)
    alpha_tr = learning_rate_decay(alpha, decay_rate, global_step, 1)

    # train
    train_op = Adam_op(loss, alpha_tr, beta1, beta2, epsilon)

    # add to graph’s collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    # Create a saver.
    saver = tf.train.Saver()
    # initialize variables
    init = tf.global_variables_initializer()

    # define number of steps
    steps = round(Data_train[0].shape[0] / batch_size)
    length = Data_train[0].shape[0]

    with tf.Session() as session:
        session.run(init)
        for epoch in range(epochs + 1):
            feed_dict = {x: Data_train[0], y: Data_train[1]}
            # train values
            t_accur = session.run(accuracy, feed_dict)
            t_loss = session.run(loss, feed_dict)
            # valid values
            vaccur = session.run(accuracy, feed_dict={x: Data_valid[0],
                                                      y: Data_valid[1]})
            v_loss = session.run(loss, feed_dict={x: Data_valid[0],
                                                  y: Data_valid[1]})
            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(t_loss))
            print('\tTraining Accuracy: {}'.format(t_accur))
            print('\tValidation Cost: {}'.format(v_loss))
            print('\tValidation Accuracy: {}'.format(vaccur))
            if epoch != epochs:
                # pointer where
                start = 0
                end = batch_size
                # shuffle training data before each epoch
                X_trainS, Y_trainS = shuffle_data(Data_train[0], Data_train[1])
                for step in range(1, steps + 2):
                    # slice train data according to mini bach size every
                    train_batch = X_trainS[start:end]
                    train_label = Y_trainS[start:end]
                    feed_dict = {x: train_batch, y: train_label}
                    # run train operation
                    # b_train = session.run(train_op, feed_dict)
                    if step % 100 == 0:
                        # compute cost and accuracy every 100 steps
                        b_cost = session.run(loss, feed_dict)
                        b_accuracy = session.run(accuracy, feed_dict)
                        print('\tStep {}:'.format(step))
                        print('\t\tCost: {}'.format(b_cost))
                        print('\t\tAccuracy: {}'.format(b_accuracy))
                    # increment point to slice according to batch size
                    start = start + batch_size
                    if (length - start) < batch_size:
                        end = end + (length - start)
                    else:
                        end = end + batch_size
            # increment global_step for learning decay and train ops
            session.run(increment_global_step)
        return saver.save(session, save_path)
