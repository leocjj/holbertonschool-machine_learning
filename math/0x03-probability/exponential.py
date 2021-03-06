#!/usr/bin/env python3
""" 0x03. Probability """


class Exponential:
    """  class that represents an Exponential distribution: """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        :param data:  list of the data to be used to estimate the distribution
        :param lambtha: expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.__lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.__lambtha = 1 / (sum(data) / len(data))

    @property
    def lambtha(self):
        """  getter method """
        return self.__lambtha

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        :param x: is the time period
        :return: the PDF value for x
        """
        if x < 0:
            return 0
        return self.__lambtha * pow(self.e, - self.__lambtha * x)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        :param x: is the time period
        :return: the PDF value for x
        """
        if x < 0:
            return 0
        return 1 - pow(self.e, - self.__lambtha * x)
