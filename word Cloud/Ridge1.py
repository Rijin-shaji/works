# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Example data
np.random.seed(42)
X = np.random.randn(100, 3)
y = 4*X[:,0] + 3*X[:,1] + 2*X[:,2] + np.random.randn(100) * 0.5

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Ridge model
ridge = Ridge(alpha=1.0)   # alpha = λ (regularization strength)

# Train model
ridge.fit(X_train, y_train)

# Predict
y_pred = ridge.predict(X_test)

# Evaluate
print("R^2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("Coefficients:", ridge.coef_)
print("Intercept:", ridge.intercept_)

# Visualize predicted vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge Regression: Actual vs Predicted")
plt.show()
