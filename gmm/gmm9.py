import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

df = pd.read_excel("F:/Pumpkin_Seeds_Dataset.xlsx")

X = df[[ 'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
        'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity',
        'Extent', 'Roundness', 'Aspect_Ration', 'Compactness']]
y = df['Class']

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

labels = gmm.predict(X)

means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

print(f"Learned Means:\n{means}\n")
print(f"Learned Covariances (first component):\n{covariances[0]}\n")
print(f"Learned Weights:\n{weights}\n")

X_pca = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, edgecolor='k')

means_pca = PCA(n_components=2).fit(X).transform(means)
plt.scatter(means_pca[:, 0], means_pca[:, 1], marker='X', s=200, c='red', label='GMM Centers')

plt.title("GMM Clustering on Breast Cancer Dataset")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.show()