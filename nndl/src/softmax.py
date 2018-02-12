from math import exp
import numpy as np
z_L = np.array([1, -1, 100, 10, -5])


def softmax(c):
    numerator = np.exp(z_L)
    denorm = sum(numerator)

    return numerator / denorm


print(softmax(1))
print(softmax(10))
print(softmax(100))
print(softmax(1000000000000000000000000000000000000000000000000000000))
