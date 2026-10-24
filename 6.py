# 6.
# Implement K-Means clustering on any dataset from repository of data and
# perform following operations.
# i. Preprocess the dataset (normalize features).
# ii. Perform clustering and visualize clusters.
# iii. Compare results of different clustering techniques.

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

data = load_iris()
X = data.data
X = MinMaxScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
k = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=k, cmap='viridis')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

hierarchy = AgglomerativeClustering(n_clusters=3)
h = hierarchy.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=h, cmap='rainbow')
plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("K-Means Score:", silhouette_score(X, k))
print("Hierarchical Score:", silhouette_score(X, h))