#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

names = ("Farrah", "Fred", "Felicia")
colors = ("red", "yellow", "#ff8000", "#ffe5b4")
base = [0, 0, 0]

for i in range(len(fruit)):
    plt.bar(
        names,
        fruit[i],
        color=colors[i],
        width=0.5,
        bottom=base
    )
    base += fruit[i]

plt.ylabel("Quantity of Fruit")
plt.yticks(range(0, 81, 10))
plt.title("Number of Fruit per Person")
plt.legend(["apples", "bananas", "oranges", "peaches"])

plt.show()
plt.close()
