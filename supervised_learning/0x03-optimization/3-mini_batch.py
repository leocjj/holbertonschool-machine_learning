#!/usr/bin/env python3
""" 0x03. Optimization """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model using mini-batch gradient descent
    :param X_train: ndarray of shape (m, 784) containing the training data
        m is the number of data points. 784 is the number of input features
    :param Y_train: one-hot numpy.ndarray of shape (m, 10) containing the
        training labels. 10 is the number of classes the model should classify
    :param X_valid: ndarray of shape (m, 784) containing the validation data
    :param Y_valid: a one-hot numpy.ndarray of shape (m, 10) containing the
        validation labels
    :param batch_size: number of data points in a batch
    :param epochs: number of times the training should pass through the whole
        dataset
    :param load_path: path from which to load the model
    :param save_path: path to where the model should be saved after training
    :return: path where the model was saved
    """

    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(session, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]

        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        steps = round(len(X_train) / batch_size)
        length = X_train.shape[0]

        for epoch in range(epochs + 1):
            temp_dict = {x: X_train, y: Y_train}

            t_accur = session.run(accuracy, temp_dict)
            t_loss = session.run(loss, temp_dict)
            vaccur = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            v_loss = session.run(loss, feed_dict={x: X_valid, y: Y_valid})

            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(t_loss))
            print('\tTraining Accuracy: {}'.format(t_accur))
            print('\tValidation Cost: {}'.format(v_loss))
            print('\tValidation Accuracy: {}'.format(vaccur))

            if epoch != epochs:
                start = 0
                end = batch_size

                X_trainS, Y_trainS = shuffle_data(X_train, Y_train)
                for step in range(1, steps + 2):
                    train_batch = X_trainS[start:end]
                    train_label = Y_trainS[start:end]
                    temp_dict = {x: train_batch, y: train_label}

                    b_train = session.run(train_op, temp_dict)
                    if step % 100 == 0:
                        b_cost = session.run(loss, temp_dict)
                        b_accuracy = session.run(accuracy, temp_dict)
                        print('\tStep {}:'.format(step))
                        print('\t\tCost: {}'.format(b_cost))
                        print('\t\tAccuracy: {}'.format(b_accuracy))

                    start = start + batch_size
                    if (length - start) < batch_size:
                        end = end + (length - start)
                    else:
                        end = end + batch_size

        return saver.save(session, save_path)
