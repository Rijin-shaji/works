import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = pd.read_csv("F:/student-mat.csv", sep=";") 

X = data.drop('G3', axis=1)  
y = data['G3']
    

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

feature_name = X.columns[0] 
plt.figure(figsize=(8,8))

plt.scatter(X_test[feature_name], y_test, label='Actual', alpha=0.7)
plt.scatter(X_test[feature_name], y_pred, label='Predicted', alpha=0.7)

plt.title("KNN Regression: Actual vs Predicted")
plt.xlabel("feature_name")
plt.ylabel("Target Value")
plt.legend()

plt.show()
