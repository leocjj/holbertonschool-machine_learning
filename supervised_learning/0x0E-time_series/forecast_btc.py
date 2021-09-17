#!/usr/bin/env python3
'''forecast btc '''
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

preprocess = __import__('preprocess_data').pre_process

x_train, y_train, x_test, y_test, sc = preprocess()

model = K.models.Sequential()

# Four LSTM layers with dropout
model.add(K.layers.LSTM(units=50, return_sequences=True,
                        input_shape=(x_train.shape[1], 1)))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.LSTM(units=50, return_sequences=True))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.LSTM(units=50, return_sequences=True))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.LSTM(units=50))
model.add(K.layers.Dropout(0.2))

model.add(K.layers.Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(100).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(32)

model.fit(train_dataset, epochs=10, steps_per_epoch=10)
model.evaluate(test_dataset, steps=10)

predicted_price = model.predict(test_dataset, steps=470)
predicted_btc = sc.inverse_transform(predicted_price)
y = sc.inverse_transform(y_test)

plt.plot(y, color='red', label='Real BTC Price')
plt.plot(predicted_btc, color='blue', label='Predicted BTC Price')
plt.title('BTC Price Prediction')
plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.legend()
plt.show()
