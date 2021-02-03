#!/usr/bin/env python3
""" 0x03. Probability """


class Normal:
    """  class that represents a normal distribution: """

    e = 2.7182818285
    pi = 3.1415926536

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

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        :param x: is the x-value
        :return: the z-score of x
        """
        return (x - self.__mean) / self.__stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        :param z: is the z-score
        :return: the x-value of z
        """
        return self.__stddev * z + self.__mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        :param x:
        :return:
        """
        return pow(self.e, -0.5 * pow(self.z_score(x), 2))\
            / (self.__stddev * pow(2 * self.pi, 0.5))

    def erf(self, x):
        return (2 / pow(self.pi, 0.5)) * (x - pow(x, 3) / 3 + pow(x, 5) / 10
                                          - pow(x, 7) / 42 + pow(x, 9) / 216)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        :param x: is the x-value
        :return: the CDF value for x
        """
        return 0.5 * (1 + self.erf(self.z_score(x) / pow(2, 0.5)))
