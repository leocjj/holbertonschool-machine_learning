#!/usr/bin/env python3
""" 0x03. Probability """


class Normal:
    """  class that represents a normal distribution: """

    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        :param data: list of the data to be used to estimate the distribution
        :param mean: is the mean of the distribution
        :param stddev: is the standard deviation of the distribution
        """
        if data is None:
            self.__mean = float(mean)
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.__stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.__mean = sum(data) / len(data)
            self.__stddev = pow(
                sum([pow(data[i] - self.__mean, 2)
                     for i in range(0, len(data))]) / (len(data)), 0.5)

    @property
    def mean(self):
        """  getter method """
        return self.__mean

    @property
    def stddev(self):
        """  getter method """
        return self.__stddev
