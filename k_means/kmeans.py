import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
n_samples = 300
income = np.random.normal(50000, 15000, n_samples)
spending_score = np.random.normal(50, 15, n_samples)

# Combine features into a single array
X = np.column_stack((income, spending_score))

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Visualize the results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using K-Means Clustering')

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)

plt.show()

# Print cluster information
for i in range(n_clusters):
    cluster_points = X[cluster_labels == i]
    print(f"Cluster {i + 1}:")
    print(f"  Number of customers: {len(cluster_points)}")
    print(f"  Average Income: ${cluster_points[:, 0].mean():.2f}")
    print(f"  Average Spending Score: {cluster_points[:, 1].mean():.2f}")
    print()