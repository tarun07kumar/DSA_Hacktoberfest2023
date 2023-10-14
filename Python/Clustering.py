#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Build and fit K-Means model
model = KMeans(n_clusters=4)
model.fit(X)

# Get cluster assignments and cluster centers
labels = model.labels_
centers = model.cluster_centers_

# Visualize data and clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()
