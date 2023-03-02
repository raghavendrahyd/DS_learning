import numpy as np
from sklearn.neighbors import NearestNeighbors


class KMeans:
    """
    KMeans clustering algorithm
    'centroids' argument can be None for usual Kmeans or "kmeans++" for kmeans++ algorithm
    """

    def __init__(self, k, centroids=None, max_iter=30, tolerance=1e-2):
        self.k = k
        self.centroids = centroids
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.labels = None
        self.clusters = None

    def fit(self, X):
        idx_s = np.random.choice(X.shape[0], self.k)
        if self.centroids == "kmeans++":
            self.centroids = select_centroids(X, self.k)
        else:
            self.centroids = X[idx_s, :]
        print(f"first random selection of centriods are the points :{idx_s}")
        iter_ = 1

        self.clusters = dict()
        while iter_ <= self.max_iter:
            self.clusters = dict()  # initiating clusters dictionary
            for idx, i in enumerate(X):
                cluster_num = np.argmin(
                    list(
                        map(lambda x: np.linalg.norm((i - x)), self.centroids)
                    )
                )  # cluster to which point belongs,check
                self.clusters.setdefault(cluster_num, set()).add(idx)

            new_centroids = np.zeros_like(self.centroids)
            for i in self.clusters.items():
                new_centroids[i[0]] = np.average(X[list(i[1]), :], axis=0)

            if (
                np.linalg.norm(self.centroids - new_centroids, axis=1).mean()
                <= self.tolerance
            ):
                break
            self.centroids = new_centroids
            iter_ = iter_ + 1
        self.labels = []

        print(
            f"ran for {iter_} iterations and created {len(self.clusters)} clusters with sizes: {list(map(lambda x: len(x[1]),self.clusters.items()))}"
        )
        for i in X:
            self.labels.append(
                np.argmin(
                    list(
                        map(lambda x: np.linalg.norm((i - x)), self.centroids)
                    )
                )
            )

    def predict(self, X):
        labels = []
        for i in X:
            labels.append(
                np.argmin(
                    list(
                        map(lambda x: np.linalg.norm((i - x)), self.centroids)
                    )
                )
            )
        return labels


def select_centroids(X, k):  # For kmeans++
    centriods = X[np.random.choice(X.shape[0], 1)].reshape(
        1, -1
    )  # first of K randomly picked
    count_k = 1
    while count_k < k:
        probs = np.zeros(X.shape[0])
        for idx, i in enumerate(X):
            probs[idx] = np.array(
                list(map(lambda x: np.linalg.norm((i - x)), centriods))
            ).min()
        probs = probs / np.sum(
            probs
        )  # probabilities proportional to the point's distance from closest centroid
        tmp_c = np.array(X[np.random.choice(X.shape[0], p=probs)]).reshape(
            1, -1
        )
        count_k = count_k + 1
        centriods = np.vstack((centriods, tmp_c))
    return centriods


# Spectral Clustering
def spectral_clustering(X: np.ndarray, k: int, graph_type: str = "knn"):
    """
    Spectral Clustering algorithm
    graph_type can be "knn" or "full"
    """
    if graph_type == "knn":
        knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
        knn.fit(X)
        A = knn.kneighbors_graph(X).toarray()
    elif graph_type == "full":
        A = np.ones((X.shape[0], X.shape[0]))

    D = np.diag(np.sum(A, axis=1))
    L = D - A

    eig_vals, eig_vecs = np.linalg.eig(L)
    eig_vals, eig_vecs = eig_vals.real, eig_vecs.real

    idx = eig_vals.argsort()[:k]
    eig_vals = eig_vals[idx]
    U = eig_vecs[:, idx]
    U = U / np.linalg.norm(U, axis=1).reshape(-1, 1)

    kmeans_obj = KMeans(k=k, centroids="kmeans++", max_iter=30, tolerance=1e-2)
    kmeans_obj.fit(U)
    labels = kmeans_obj.labels
    centroids = kmeans_obj.centroids

    return centroids, labels
