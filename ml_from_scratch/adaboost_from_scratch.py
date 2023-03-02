# Adaboost from scratch
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, num_iter, max_depth=1):
        self.num_iter = num_iter
        self.max_depth = max_depth
        self.trees = []
        self.trees_weights = []

    def fit(self, X, y):
        N, _ = X.shape
        d = np.ones(N) / N
        for m in range(self.num_iter):
            tree_i = DecisionTreeClassifier(max_depth=self.max_depth)
            # using decision tree from sklearn because it has a sample_weight parameter
            tree_i.fit(X, y, sample_weight=d)
            self.trees.append(tree_i)
            y_pred = tree_i.predict(X)
            error_points = y_pred != y
            err_m = np.max(
                [np.multiply(d, error_points).sum() / np.sum(d), 0.003]
            )
            # added a lower cap in err_m to evade div by zero issue
            alpha_m = np.log((1 - err_m) / err_m)
            self.trees_weights.append(alpha_m)
            d = np.multiply(np.exp(np.multiply(alpha_m, error_points)), d)

    def predict(self, X):
        N, _ = X.shape
        y = np.zeros(N)
        for i in range(N):
            tmp_y_i = sum(
                [
                    self.trees_weights[idx] * (j.predict(X[i].reshape(1, -1)))
                    for idx, j in enumerate(self.trees)
                ]
            )
            y[i] = np.sign(tmp_y_i)
        return y
