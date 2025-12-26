import numpy as np

hours = np.array([2, 4, 6, 8, 9])
passed = np.array([0, 0, 1, 1, 1])

w = 0.62
b = -3.57

z = w * hours + b
p = 1 / (1 + np.exp(-z)) 


for x, y, prob in zip(hours, passed, p):
    print(f"Hour = {x}, Passed = {y}, Predicted P = {prob:.3f}")
