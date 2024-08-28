import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# This example demonstrates how to use PCA for dimensionality reduction and visualization using the Iris dataset. Here's a breakdown of what the code does:

# We import the necessary libraries and load the Iris dataset.
# We standardize the features using StandardScaler to ensure all features are on the same scale.
# We apply PCA to the scaled data.
# We create two visualizations:
# a. A plot showing the cumulative explained variance ratio vs. the number of components.
# b. A scatter plot of the first two principal components, color-coded by iris species.
# Finally, we print the explained variance ratio for each principal component.

# The visualizations help us understand:

# How many principal components are needed to explain a certain percentage of the variance in the data.
# How well the first two principal components separate the different iris species.

# This example showcases how PCA can be used for:

# Dimensionality reduction: By choosing a subset of principal components, we can reduce the number of features while retaining most of the information.
# Data visualization: By projecting the data onto the first two or three principal components, we can visualize high-dimensional data in 2D or 3D plots.
# Feature importance: The explained variance ratio gives us an idea of how much information each principal component captures.

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate the cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance ratio
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

# Plot the first two principal components
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b']
for i, c in zip(range(3), colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=c, label=iris.target_names[i])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Iris Dataset - First Two Principal Components')
plt.legend()
plt.grid(True)
plt.show()

# Print the explained variance ratio for each component
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1} explained variance ratio: {ratio:.4f}")