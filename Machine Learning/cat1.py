import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt

# ---------- Load data ----------
# Assumes 'data.csv' with features and a target column named 'target'
df = pd.read_csv("data.csv")

# Quick handling: drop rows with missing target
df = df.dropna(subset=['target'])

# ---------- Prepare X, y ----------
y = df['target']
X = df.drop(columns=['target'])

# ---------- Identify categorical columns ----------
# Two common options:
# 1) If dtype object or category:
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 2) Or if you know names: cat_cols = ['col1','col2']
# Use whichever method suits your dataset.

# If there are missing values in cat columns, fill with placeholder (CatBoost can handle missing numerics)
for c in cat_cols:
    X[c] = X[c].astype('str').fillna('NA')  # convert to str to be safe

# ---------- Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None
)

# ---------- Create CatBoost Pool (optional but useful) ----------
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
test_pool  = Pool(data=X_test,  label=y_test,  cat_features=cat_cols)

# ---------- Instantiate model ----------
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='Accuracy',      # or 'Logloss' for probability-focused
    early_stopping_rounds=50,
    verbose=100,
    random_seed=42
)

# ---------- Train ----------
model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# ---------- Predict & evaluate ----------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] if model.is_fitted() and model.get_params()['loss_function'].startswith('Logloss') else None

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Feature importance ----------
fi = model.get_feature_importance(train_pool)
feat_names = X_train.columns
# simple bar plot
plt.figure(figsize=(8,6))
plt.barh(feat_names, fi)
plt.xlabel("Importance")
plt.title("Feature importance (CatBoost)")
plt.tight_layout()
plt.show()

# ---------- Save / Load ----------
model.save_model("catboost_classifier.cbm")
# To load:
# from catboost import CatBoostClassifier
# model2 = CatBoostClassifier()
# model2.load_model("catboost_classifier.cbm")
