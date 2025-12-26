import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("F:/iris.csv", encoding='latin-1')
X = df[['x0','x1','x2','x3','x4']]

# -----------------------------
# 2. Standardize
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Fit K-Means
# -----------------------------
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# -----------------------------
# 4. Silhouette Score
# -----------------------------
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score for K={k}: {sil_score:.4f}")

# -----------------------------
# 5. Plot clusters (first two features)
# -----------------------------
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan']
markers = ['o', 's', '^', 'D', 'P', 'X']

plt.figure(figsize=(8,6))

for cluster in range(k):
    cluster_points = X_scaled[labels == cluster]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=70,
        c=colors[cluster],
        marker=markers[cluster],
        label=f"Cluster {cluster+1}"
    )

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=300,
    c='red',
    marker='X',
    edgecolors='black',
    linewidths=2,
    label='Centroids'
)

plt.title(f"K-Means Clustering with Silhouette Score={sil_score:.4f}")
plt.xlabel("x0 (scaled)")
plt.ylabel("x1 (scaled)")
plt.legend()
plt.grid(True)
plt.show()
