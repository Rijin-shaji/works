import numpy as np
from sklearn.linear_model import Perceptron

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 1])  

model = Perceptron(
    max_iter=1000,
    eta0=0.1
)

model.fit(X, y)
predictions = model.predict(X)

print("OR Gate Results:")
for inp, pred in zip(X, predictions):
    print(f"{inp[0]} OR {inp[1]} = {pred}")

print("\nModel Parameters:")
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
