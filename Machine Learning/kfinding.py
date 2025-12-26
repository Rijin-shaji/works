from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

df = pd.read_csv("F:/iris.csv", encoding='latin-1')

X = df[['x0', 'x1', 'x2', 'x3', 'x4']]
y = df['type']

def best_k_by_silhouette(X, k_min=2, k_max=10):

    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):

        kmeans = KMeans(n_clusters=k, random_state=42)   
        labels = kmeans.fit_predict(X)                  

        score = silhouette_score(X, labels)

        print(f"K={k}, Silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print("\nBest K =", best_k)
    return best_k

print(best_k_by_silhouette(X))
