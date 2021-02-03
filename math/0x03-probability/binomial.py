#!/usr/bin/env python3
""" 0x03. Probability """


class Binomial:
    """  class that represents a Binomial distribution: """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """
        :param data: list of the data to be used to estimate the distribution
        :param n: is the number of Bernoulli trials
        :param p: is the probability of a “success”
        """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if not (0 < p < 1):
                raise ValueError('p must be greater than 0 and less than 1')
            self.__n = int(n)
            self.__p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            m = sum(data) / len(data)
            self.__p = 1 - (sum([pow(data[i] - m, 2)
                                 for i in range(len(data))]) / len(data)) / m
            self.__n = int(round(m / self.p))
            self.__p = m / self.n

    @property
    def n(self):
        """  getter method """
        return self.__n

    @property
    def p(self):
        """  getter method """
        return self.__p

    @staticmethod
    def fact(k):
        fact_k = 1
        for i in range(1, k + 1):
            fact_k *= i
        return fact_k

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        :param k: is the number of “successes”
        :return: the PMF value for k
        """
        if not (0 <= k <= self.__n):
            return 0
        k = int(k)
        C = self.fact(self.__n) / (self.fact(k) * self.fact(self.__n - k))
        return C * pow(self.__p, k) * pow(1 - self.__p, self.__n - k)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        :param k: is the number of “successes”
        :return: the CDF value for k
        """
        if not (0 <= k <= self.__n):
            return 0
        k = int(k)
        return sum([self.pmf(i) for i in range(k + 1)])
