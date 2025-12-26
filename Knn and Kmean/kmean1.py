import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("F:/iris.csv", encoding='latin-1')

X = df[['x1','x2','x3','x4']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels

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

centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=300,
    c='red',
    marker='X',
    edgecolors='black',
    linewidths=2,
    label="Centroids"
)

plt.title("K-Means Clustering (Iris dataset)")
plt.xlabel("x0 (scaled)")
plt.ylabel("x1 (scaled)")
plt.legend()
plt.grid(True)
plt.show()


wcss = kmeans.inertia_
print(f"WCSS for K={k}: {wcss}")
