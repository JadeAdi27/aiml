import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('ch1ex1.csv')
points = df.values

# KMeans clustering
model = KMeans(n_clusters=3)
model.fit(points)
labels = model.predict(points)

# Extract cluster centroids
centroids = model.cluster_centers_
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Scatter plot
xs = points[:, 0]
ys = points[:, 1]
plt.scatter(xs, ys, c=labels)
plt.scatter(centroids_x, centroids_y, marker='X', s=200, c='red')
plt.show()
