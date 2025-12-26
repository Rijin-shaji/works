import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
df = pd.read_csv("F:/telecom_churn_403.csv", encoding='latin-1')
df=df.dropna()

df = pd.get_dummies(df, drop_first=True)

# ----------------------------
# Split features & labels
# ----------------------------
X = df.drop("Next_Month_Data_Usage", axis=1)
y = df["Next_Month_Data_Usage"]


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
# Train SVR (Regression)
# ----------------------------
model = SVR(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ----------------------------
# Train REGRESSION model on 2D projection for visualization
# ----------------------------
model_2d = SVR(kernel='rbf', C=1.0, gamma='scale')
model_2d.fit(X_train_2d, y_train)

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
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
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
plt.title("SVR Regression Surface (2D SVD Projection)")
plt.show()

