import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# ----------------------------
# Load dataset
# ----------------------------
df=pd.read_csv("F:/house_price_data.csv")
X=df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement']]
y=df['price']

# One-hot encoding for categorical columns
X = pd.get_dummies(X, drop_first=True)

# ----------------------------
# 2D Projection (Truncated SVD)
# ----------------------------
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X)

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test, X_train_2d, X_test_2d = train_test_split(
    X, y, X_2d, test_size=0.2, random_state=42
)

# ----------------------------
# Train XGBoost Regressor
# ----------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Metrics
print("\nMSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ----------------------------
# Train XGBoost on 2D projection for visualization
# ----------------------------
xgb_2d = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=3,  # smaller depth for 2D
    learning_rate=0.1,
    random_state=42
)
xgb_2d.fit(X_train_2d, y_train)

# ----------------------------
# Create contourf grid
# ----------------------------
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Predict continuous values on grid
Z = xgb_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ----------------------------
# Plot regression surface
# ----------------------------
plt.figure(figsize=(8, 6))
cp = plt.contourf(xx, yy, Z, alpha=0.7, cmap=plt.cm.coolwarm)

# scatter actual training samples
plt.scatter(
    X_train_2d[:, 0],
    X_train_2d[:, 1],
    c=y_train,
    cmap=plt.cm.coolwarm,
    edgecolor='k',
    s=50
)

plt.colorbar(cp, label="Predicted Price")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("XGBoost Regression Surface (2D SVD Projection)")
plt.show()