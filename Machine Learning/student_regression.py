import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("F:/student-mat.csv", sep=";") 

X = data.drop('G3', axis=1)  
y = data['G3']
    
X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lasso = LassoCV(cv=5, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("Lasso Best alpha:", lasso.alpha_)
print("Lasso MSE:", mean_squared_error(y_test, y_pred_lasso))
print("Lasso R2:", r2_score(y_test, y_pred_lasso))
print("------------")


ridge_alphas = np.logspace(-4, 4, 50)  
ridge = RidgeCV(alphas=ridge_alphas, cv=5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("Ridge Best alpha:", ridge.alpha_)
print("Ridge MSE:", mean_squared_error(y_test, y_pred_ridge))
print("Ridge R2:", r2_score(y_test, y_pred_ridge))
print("------------")

elastic = ElasticNetCV(l1_ratio=[0.1,0.5,0.7,0.9,0.95,1], cv=5, max_iter=10000)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)
print("Elastic Net Best alpha:", elastic.alpha_)
print("Elastic Net Best l1_ratio:", elastic.l1_ratio_)
print("Elastic Net MSE:", mean_squared_error(y_test, y_pred_elastic))
print("Elastic Net R2:", r2_score(y_test, y_pred_elastic))

models = ['Lasso', 'Ridge', 'ElasticNet']
mse_values = [mean_squared_error(y_test, y_pred_lasso),
              mean_squared_error(y_test, y_pred_ridge),
              mean_squared_error(y_test, y_pred_elastic)]


plt.figure(figsize=(8,5))
plt.plot(models, mse_values, marker='o', linestyle='-', color='blue', linewidth=2)
plt.ylabel('Mean Squared Error')
plt.title('Model Comparison based on MSE')
plt.grid(True)
plt.show()