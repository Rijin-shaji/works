import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# 1. Load Your Dataset
df = pd.read_csv("F:/data.csv", encoding='latin-1')

X = df[[ 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
        'symmetry_mean', 'fractal_dimension_mean']]
y = df['diagnosis']

y = y.map({'M': 1, 'B': 0})

# 3. Scale (recommended for SVD)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply Truncated SVD
svd = TruncatedSVD(n_components=5, random_state=42) 
X_reduced = svd.fit_transform(X_scaled)

print("Original shape:", X_scaled.shape)
print("Reduced shape:", X_reduced.shape)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.25, random_state=42
)

# 6. Train a Classifier
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# 7. Predictions & Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy :", acc)
print("\nConfusion Matrix:\n", cm)