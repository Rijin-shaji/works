from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("F:/iris.csv", encoding='latin-1')
X = df[['x1', 'x2', 'x3', 'x4']]

# Elbow (WCSS) and Silhouette
wcss = []
sil_scores = []
k_range = list(range(2, 11)) 

best_k_sil = 2
best_sil = -1

print("K | Silhouette Score | WCSS")
print("-----------------------------------")

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    # WCSS
    wcss.append(kmeans.inertia_)

    # Silhouette Score
    sil = silhouette_score(X, labels)
    sil_scores.append(sil)

    print(f"{k} | {sil:.4f} | {kmeans.inertia_:.2f}")

    # Track best silhouette
    if sil > best_sil:
        best_sil = sil
        best_k_sil = k

print("\nBest K according to Silhouette Score =", best_k_sil)


# Elbow Method (distance from line)

x = np.arange(2, 11)
y = np.array(wcss)
p1 = np.array([x[0], y[0]])
p2 = np.array([x[-1], y[-1]])

distances = []
for i in range(len(x)):
    p = np.array([x[i], y[i]])
    distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
    distances.append(distance)

elbow_k = x[np.argmax(distances)]
print("Best K according to Elbow Method =", elbow_k)

# Combine Elbow & Silhouette

if abs(best_k_sil - elbow_k) <= 1:
    final_k = best_k_sil
else:
    final_k = int(round((best_k_sil + elbow_k)/2))

print("\n================ FINAL RESULTS ================")
print(f"Best K by Silhouette Score = {best_k_sil}")
print(f"Best K by Elbow Method     = {elbow_k}")
print(f"Final Combined Best K      = {final_k}")
print("================================================\n")