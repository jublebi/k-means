from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Algorithms:
    def __init__(self, X):
        self.X = X

    def KMeansAlgorithm(self, maxIteration, clusters):
        yKMeans = []
        centers = []
        output = []

        for i in range(1, maxIteration + 1):

            if i == 1:
                kmeans = KMeans(n_clusters = clusters, init = 'random', max_iter = i, n_init = 10, random_state = 0)
                y_kmeans = kmeans.fit_predict(self.X)
                yKMeans.append(y_kmeans)
                centers.append(kmeans.cluster_centers_)
                startCentroid = np.array(centers, np.int32)

            else:
                kmeans = KMeans(n_clusters = clusters, init = startCentroid[0], max_iter = i, n_init = 1,
                                random_state = 0)
                y_kmeans = kmeans.fit_predict(self.X)
                yKMeans.append(y_kmeans)
                centers.append(kmeans.cluster_centers_)

        output.append(yKMeans)
        output.append(centers)
        return output


dataset = load_iris()
X = dataset.data

clusters = 3
maxIter = 100

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2)
fig.set_size_inches(18.5, 10.5)

axes = {
    1: ax1,
    2: ax2,
    3: ax3,
    4: ax4,
    5: ax5,
    6: ax6,
    7: ax7,
    8: ax8
}

algorithm = Algorithms(X)
result = algorithm.KMeansAlgorithm(maxIter, clusters)
yKmeans = result[0]
centers = result[1]


def fill_ScatterPlot(index, Iteration):
    index = int(index)
    axes[index].scatter(X[yKmeans[Iteration - 1] == 0, 0], X[yKmeans[Iteration - 1] == 0, 1], s = 25, c = 'red',
                        label = 'Cluster 1')
    axes[index].scatter(X[yKmeans[Iteration - 1] == 1, 0], X[yKmeans[Iteration - 1] == 1, 1], s = 25, c = 'blue',
                        label = 'Cluster 2')
    axes[index].scatter(X[yKmeans[Iteration - 1] == 2, 0], X[yKmeans[Iteration - 1] == 2, 1], s = 25, c = 'green',
                        label = 'Cluster 3')
    axes[index].scatter(X[yKmeans[Iteration - 1] == 3, 0], X[yKmeans[Iteration - 1] == 3, 1], s = 25, c = 'cyan',
                        label = 'Cluster 4')
    axes[index].scatter(X[yKmeans[Iteration - 1] == 4, 0], X[yKmeans[Iteration - 1] == 4, 1], s = 25, c = 'magenta',
                        label = 'Cluster 5')
    axes[index].scatter(centers[Iteration - 1][:, 0], centers[Iteration - 1][:, 1], s = 75, c = 'black',
                        label = 'Centroids', marker = 's', alpha = 0.4)
    axes[index].set_title("Iteration: " + str(Iteration))


for i in range(len(yKmeans)):
    iteration = i + 1

    if iteration <= 4:
        fill_ScatterPlot(iteration, iteration)

    elif (iteration % 10) == 0 and (iteration / 10 + 4) <= 7:
        dictIndex = iteration / 10 + 4
        fill_ScatterPlot(dictIndex, iteration)

    elif iteration == maxIter:
        dictIndex = 8
        fill_ScatterPlot(dictIndex, iteration)

plt.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.97)
plt.show()



