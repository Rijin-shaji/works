import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools

# Load Dataset
df = pd.read_csv("F:/car_data.csv") 

X = df.drop('High_Price', axis=1)

# 2️⃣ Standardize Features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ K-Means Example Plot (k=3)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

colors = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan']
markers = ['o', 's', '^', 'D', 'P', 'X']

colors_cycle = itertools.cycle(colors)
markers_cycle = itertools.cycle(markers)

plt.figure(figsize=(8,6))
for cluster in range(k):
    cluster_points = X_scaled[labels == cluster]  
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=70,
        c=next(colors_cycle),
        marker=next(markers_cycle),
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

plt.title("K-Means Clustering (k=3)")
plt.xlabel("x0 (scaled)")
plt.ylabel("x1 (scaled)")
plt.legend()
plt.grid(True)
plt.show()



# 4️⃣ Silhouette Score

sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    print(f"Silhouette Score for K={k}: {sil:.4f}")

best_silhouette_k = range(2, 11)[sil_scores.index(max(sil_scores))]
print(f"Best K according to Silhouette Score: {best_silhouette_k}")



# 5️⃣ Gap Statistic Function

def gap_statistic(X, refs=10, max_k=10):
    gaps = np.zeros(max_k)
    for k in range(1, max_k+1):
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        disp = km.inertia_

        ref_disps = np.zeros(refs)
        for i in range(refs):
            X_ref = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), X.shape)
            km_ref = KMeans(n_clusters=k, random_state=42).fit(X_ref)
            ref_disps[i] = km_ref.inertia_

        gaps[k-1] = np.log(np.mean(ref_disps)) - np.log(disp)
    return gaps

gaps = gap_statistic(X_scaled, refs=10, max_k=10)
gap_k = gaps.argmax() + 1
print(f"Optimal K according to Gap Statistic: {gap_k}")



# 6️⃣ K-Means with Gap Statistic K

k_final = gap_k
kmeans = KMeans(n_clusters=k_final, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

colors_cycle = itertools.cycle(colors)
markers_cycle = itertools.cycle(markers)

plt.figure(figsize=(8,6))
for cluster in range(k_final):
    cluster_points = X_scaled[labels == cluster]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=70,
        c=next(colors_cycle),
        marker=next(markers_cycle),
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
    label='Centroids'
)

plt.title(f"K-Means Clustering with K={k_final}")
plt.xlabel("x0 (scaled)")
plt.ylabel("x1 (scaled)")
plt.legend()
plt.grid(True)
plt.show()
