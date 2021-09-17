#!/usr/bin/env python3
""" Preprocessing """
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def pre_process():
    """
    Columns in data file:
    Timestamp   Open   High   Low   Close   Volume_(BTC)   Volume_(Currency)
        Weighted_Price
    """
    filename = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    bitstamp = pd.read_csv(filename)
    bitstamp = bitstamp.drop(["High", "Low", "Open", "Volume_(BTC)",
                              "Volume_(Currency)", "Weighted_Price"], axis=1)
    bitstamp["Timestamp"] = pd.to_datetime(bitstamp['Timestamp'], unit='s')
    bitstamp = bitstamp[bitstamp["Timestamp"].dt.year >= 2018]
    bitstamp = bitstamp.resample('H', on='Timestamp').mean()
    bitstamp["Close"] = bitstamp["Close"].fillna(method="ffill")

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set = sc.fit_transform(bitstamp)
    del bitstamp
    result = []
    for index in range(len(training_set) - 24):
        result.append(training_set[index: index + 25])

    result = np.array(result)
    training_set = result[:(training_set.shape[0] // 2), :]
    test_set = result[(training_set.shape[0] // 2):, :]

    X_train = training_set[:, :-1]
    y_train = training_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return X_train, y_train, x_test, y_test, sc
