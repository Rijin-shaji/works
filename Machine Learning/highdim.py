from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import xgboost as xgb

# ------------------------------------------------------
# 1) LOAD DATASET SAFELY
# ------------------------------------------------------
df = pd.read_csv("F:/data.csv", encoding='latin-1')
df=df.dropna()

X = df[[ 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
        'symmetry_mean', 'fractal_dimension_mean']]
y = df['diagnosis']

y = y.map({'M': 1, 'B': 0})

# KEEP ONLY NUMERIC FEATURES
X = df.drop("diagnosis", axis=1).select_dtypes(include=[np.number])

if X.shape[0] == 0 or X.shape[1] == 0:
    raise ValueError("ERROR: No numeric features found. Check dataset.")

print("\nNumeric feature shape:", X.shape)

# ------------------------------------------------------
# 4) INCREASE DIMENSIONS (Polynomial)
# ------------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_expanded = poly.fit_transform(X)

print("Expanded feature shape:", X_expanded.shape)

# ------------------------------------------------------
# 5) REDUCE TO EXACTLY 30 DIMENSIONS (PCA)
# ------------------------------------------------------
n_features_expanded = X_expanded.shape[1]
n_samples = X_expanded.shape[0]

if n_samples < 2:
    raise ValueError("Not enough samples for PCA. Need at least 2 rows.")

n_components = min(30, n_features_expanded, n_samples)

pca = PCA(n_components=n_components)
X_30 = pca.fit_transform(X_expanded)

print("Final PCA shape:", X_30.shape)

# ------------------------------------------------------
# 6) Train-test split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_30, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# 7) Train XGBoost
# ------------------------------------------------------
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# ------------------------------------------------------
# 8) Evaluate
# ------------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
