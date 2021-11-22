import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('iris.data', header=None)

dataset.rename(columns={0: 'sepal_length', 1: 'sepal_width', 2: 'petal_length', 3: 'petal_width', 4: 'class'},
               inplace=True)
dataset.to_csv('iris_data.csv', index=False)

# Splitting data into x and y(label)
x = dataset.iloc[:, 0: 4].values
y = dataset.iloc[:, 4].values


class K_means:
    def __init__(self, K=3, no_of_iteration=300):
        self.clusters = {}
        self.centroids = {}
        self.K = K
        self.no_of_iteration = no_of_iteration

    # function to calculate euclidean distance
    def euclidean_distance(self, p1, p2):
        dist = np.linalg.norm(p1 - p2, axis=0)
        return dist

    # function for the clustering
    def k_means(self, data):
        # initializing first k points of the data as centroids
        for i in range(self.K):
            self.centroids[i] = data[i]

        # initializing k clusters
        for i in range(self.no_of_iteration):
            for j in range(self.K):
                self.clusters[j] = []

        # calculating distance between data points and centroids
        for d in data:
            distance = []
            for c in self.centroids:
                distance.append(self.euclidean_distance(d, self.centroids[c]))

            # assigning data points to corresponding clusters
            id_cluster = distance.index(min(distance))
            self.clusters[id_cluster].append(d)

        # recalculating new centroids
        for id_cluster in self.clusters:
            self.centroids[id_cluster] = np.average(self.clusters[id_cluster], axis=0)


# Evaluating value of k
SSE = []
K = [1, 2, 3, 5, 7, 9]

# Test out multiple values for k
for k in K:
    kmeans = K_means(k)
    kmeans.k_means(x)

    # Extracting the clusters and centroids
    final_clusters = kmeans.clusters
    final_centroids = kmeans.centroids

    # Calculating the distortion
    x4 = []
    y4 = []
    for centroid in final_centroids:
        x2 = final_centroids[centroid][0]
        y2 = final_centroids[centroid][1]
        x4.append(x2)
        y4.append(y2)

    for cluster in final_clusters:
        x3 = []
        y3 = []
        for col in final_clusters[cluster]:
            x1 = col[0]
            y1 = col[1]
            x3.append(x1)
            y3.append(y1)

    for xx in x4:
        diff_x = []
        diff_x = (xx - x3) ** 2
    for yy in y4:
        diff_y = []
        diff_y = (yy - y3) ** 2

        sse = sum(diff_y + diff_x)
    SSE.append(sse)

# plotting SSE vs value of K
plt.plot(K, SSE, 's-', markersize=5, color='red', mec='red')
plt.xlabel('Value of K')
plt.xticks(K)
plt.ylabel('SSE')
plt.title('Elbow Method for evaluating value of k')
plt.show()

print('\nDecided value of K = 3')

# no of clusters
K = 3
no_of_iteration = 300

K_Means = K_means(K)
K_Means.k_means(x)

print('\nThe final centroids are:', K_Means.centroids)
print('\nThe final clusters are:', K_Means.clusters)

# Plotting the final clusters with their centroids
colors = ["r", "g", "b"]
for id_cluster in K_Means.clusters:
    color = colors[id_cluster]
    for col in K_Means.clusters[id_cluster]:
        plt.scatter(col[0], col[1], color=color, s=10)
    for centroid in K_Means.centroids:
        color = colors[centroid]
        plt.scatter(K_Means.centroids[centroid][0], K_Means.centroids[centroid][1], color=color, s=200, marker="*")
plt.title('Final clusters and their centroids')
plt.show()
