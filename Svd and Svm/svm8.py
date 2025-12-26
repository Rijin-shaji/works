import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("F:/high_dim.csv") 

X = data.drop('target', axis=1)  
y = data['target']
    
X = pd.get_dummies(X, drop_first=True)


svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X)

# ----------------------------
# Split train/test
# ----------------------------
X_train, X_test, y_train, y_test, X_train_2d, X_test_2d = train_test_split(
    X, y, X_2d, test_size=0.2, random_state=42
)

# ----------------------------
# Train SVM on TF-IDF
# ----------------------------
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ----------------------------
# Encode labels for contour plot
# ----------------------------
le = LabelEncoder()
y_train_num = le.fit_transform(y_train)

# ----------------------------
# Train SVM on 2D projection
# ----------------------------
model_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
model_2d.fit(X_train_2d, y_train_num)

# ----------------------------
# Decision boundary (contourf)
# ----------------------------
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ----------------------------
# Plotting
# ----------------------------
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

scatter = plt.scatter(
    X_train_2d[:, 0],
    X_train_2d[:, 1],
    c=y_train_num,
    cmap=plt.cm.coolwarm,
    edgecolor='k',
    s=50
)

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("SVM Decision Boundary (TruncatedSVD 2D Projection)")

# Legend
labels = le.inverse_transform([0, 1])
handles = scatter.legend_elements()[0]
plt.legend(handles, labels)

plt.show()