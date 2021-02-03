#!/usr/bin/env python3
""" 0x03. Probability Task 10 """


class Poisson:
    """  class that represents a poisson distribution: """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
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

    @property  # Property wrap
    def lambtha(self):  # getter method
        return self.__lambtha
