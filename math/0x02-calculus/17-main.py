#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))
poly = []
print(poly_integral(poly))
poly = [0]
print(poly_integral(poly))
poly = [5]
print(poly_integral(poly))
poly = [0, 1]
print(poly_integral(poly))
poly = [3, 0, 1]
print(poly_integral(poly))
print()
poly = [5, 3, 0, 1]
print(poly_integral(poly, C=2))
poly = []
print(poly_integral(poly, C=2))
poly = [0]
print(poly_integral(poly, C=2))
poly = [5]
print(poly_integral(poly, C=2))
poly = [0, 1]
print(poly_integral(poly, C=2))
poly = [3, 0, 1]
print(poly_integral(poly, C=2))
