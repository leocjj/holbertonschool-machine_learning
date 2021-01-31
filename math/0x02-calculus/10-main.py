#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative
#      x0 x1 x2 x3
poly = [5, 3, 0, 1]
print(poly_derivative(poly))
poly = []
print(poly_derivative(poly))
poly = [0]
print(poly_derivative(poly))
poly = [5]
print(poly_derivative(poly))
poly = [-5]
print(poly_derivative(poly))
poly = [3, 4]
print(poly_derivative(poly))
poly = [3, 4, 2]
print(poly_derivative(poly))
poly = [3, 0, 2, 0]
print(poly_derivative(poly))
poly = [3, 0, 2, 0, 1]
print(poly_derivative(poly))
