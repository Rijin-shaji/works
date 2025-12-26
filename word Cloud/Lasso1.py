import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train Lasso model
lasso = Lasso(alpha=0.1)   # alpha = λ (regularization strength)
lasso.fit(X_train, y_train)

# Predict
y_pred = lasso.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Show coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lasso.coef_
})
print(coef_df)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.title("LASSO Coefficients (alpha=0.1)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)

print("Best alpha:", lasso_cv.alpha_)
print("R2 Score:", r2_score(y_test, lasso_cv.predict(X_test)))
