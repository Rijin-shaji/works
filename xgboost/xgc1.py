import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_excel("F:/uber driver.xlsx")
print(df.head(5))

# Clean column names to remove spaces/special characters
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Replace inf/-inf and drop rows with NaN in target
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['fare_amount'])

# Split features & target
X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- XGBoost Regression ---")
print("MSE:", mse)
print("R2 Score:", r2)


