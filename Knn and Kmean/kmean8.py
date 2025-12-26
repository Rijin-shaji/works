import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and scale data
df = pd.read_csv("F:/Shopping_data.csv", encoding='latin-1')
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
k = 6  
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels
centroids = kmeans.cluster_centers_

cluster_labels = {}
for i in range(k):
    # Get cluster centroid in original scale
    age_c_scaled, income_c_scaled, spend_c_scaled = centroids[i]

    # Transform scaled centroid back to original values
    age_c = age_c_scaled * X['Age'].std() + X['Age'].mean()
    income_c = income_c_scaled * X['Annual Income (k$)'].std() + X['Annual Income (k$)'].mean()
    spend_c = spend_c_scaled * X['Spending Score (1-100)'].std() + X['Spending Score (1-100)'].mean()

    # Assign segments based on actual values
    if age_c <= 18:
        cluster_labels[i] = 'Children'
    elif age_c >= 56:
        cluster_labels[i] = 'Old Age'
    elif income_c < X['Annual Income (k$)'].mean() and spend_c < X['Spending Score (1-100)'].mean():
        cluster_labels[i] = 'Low Income, Low Spending'
    elif income_c < X['Annual Income (k$)'].mean() and spend_c >= X['Spending Score (1-100)'].mean():
        cluster_labels[i] = 'Low Income, High Spending'
    elif income_c >= X['Annual Income (k$)'].mean() and spend_c < X['Spending Score (1-100)'].mean():
        cluster_labels[i] = 'High Income, Low Spending'
    else:
        cluster_labels[i] = 'High Income, High Spending'


df['Segment'] = df['Cluster'].map(cluster_labels)

# Plot all clusters together
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
        label=cluster_labels[cluster]
    )

# Plot centroids
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    s=300,
    c='red',
    marker='X',
    edgecolors='black',
    linewidths=2,
    label='Centroids'
)

plt.title("K-Means Clustering with Segments")
plt.xlabel("Age (scaled)")
plt.ylabel("Annual Income (scaled)")
plt.legend()
plt.grid(True)
plt.show()

# Separate plots for each cluster
for cluster in range(k):
    cluster_points = X_scaled[labels == cluster]
    plt.figure(figsize=(6,4))
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=70,
        c=colors[cluster],
        marker=markers[cluster]
    )
    plt.scatter(
        centroids[cluster,0],
        centroids[cluster,1],
        s=300,
        c='red',
        marker='X',
        edgecolors='black',
        linewidths=2
    )
    plt.title(f"Cluster {cluster+1}: {cluster_labels[cluster]}")
    plt.xlabel("Age (scaled)")
    plt.ylabel("Annual Income (scaled)")
    plt.grid(True)
    plt.show()


print(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster', 'Segment']].head())
summary = df.groupby('Segment')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\nCluster summary statistics:")
print(summary)
