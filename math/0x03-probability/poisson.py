#!/usr/bin/env python3
""" 0x03. Probability Task 10 """


class Poisson:
    """  class that represents a poisson distribution: """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """  init method """
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
            self.__lambtha = sum(data) / len(data)

    @property
    def lambtha(self):
        """  getter method """
        return self.__lambtha

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        :param k:  is the number of “successes”
        :return: PMF value for k
        """
        k = int(k)
        fact_k = 1
        if k < 0:
            return 0
        elif k > 0:
            for i in range(1, k + 1):
                fact_k *= i
        return pow(self.e, - self.__lambtha) * pow(self.__lambtha, k) / fact_k
