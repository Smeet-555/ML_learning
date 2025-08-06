from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load data
data = load_iris()
X = data.data

# Apply hierarchical clustering
Z = linkage(X, method='ward')  # You can try 'single', 'complete', 'average'

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
