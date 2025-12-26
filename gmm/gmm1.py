import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


df = pd.read_csv("F:/data.csv", encoding='latin-1')

X = df[[ 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
        'symmetry_mean', 'fractal_dimension_mean']]
y = df['diagnosis']

y = y.map({'M': 1, 'B': 0})


n_components = 3
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X)


labels = gmm.predict(X)
probs = gmm.predict_proba(X)  

df['Cluster'] = labels
print(df.head())

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=50)
plt.title('GMM Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
