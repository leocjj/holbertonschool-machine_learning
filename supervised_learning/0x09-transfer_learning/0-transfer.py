#!/usr/bin/env python3
""" 0x09 Tranfer Learning"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for your model:
    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
        where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
    X_p is a numpy.ndarray containing the preprocessed X
    Y_p is a numpy.ndarray containing the preprocessed Y
    """

    X = X.astype('float32')
    X /= 255
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    base_model = K.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                            pooling='avg',
                                            classes=y_train.shape[1])

    model = K.Sequential()
    model.add(K.layers.UpSampling2D())
    model.add(base_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(256, activation=('relu')))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.Dense(256, activation=('relu')))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.Dense(10, activation=('softmax')))
    callback = []


    def rate_decay(epoch):
        """ Decay for learning rate """
        return 0.001 / (1 + 1 * 30)


    learning = K.callbacks.LearningRateScheduler(schedule=rate_decay, verbose=1)
    callback.append(learning)
    callback.append(K.callbacks.ModelCheckpoint('cifar10.h5',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max'))

    model.compile(optimizer=K.optimizers.Adam(lr=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=1, batch_size=32,
                        validation_data=(x_test, y_test))

    model.save('cifar10.h5')
