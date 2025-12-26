import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("F:/house_price_data.csv")

X = df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement']]
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

param_grid = {'n_neighbors': list(range(1, 21))}
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)

best_k = grid.best_params_['n_neighbors']
print("Best n_neighbors:", best_k)

knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R² Score:", r2)

feature_idx = 2  

plt.figure(figsize=(8,6))
plt.scatter(X_test[:, feature_idx], y_test, label='Actual', alpha=0.7)
plt.scatter(X_test[:, feature_idx], y_pred, label='Predicted', alpha=0.7)
plt.xlabel(X.columns[feature_idx])
plt.ylabel("Price")
plt.title("KNN Regression: Actual vs Predicted")
plt.legend()
plt.show()

