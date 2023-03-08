import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
        self, num_iter=100, max_depth=1, nu=0.1, tol=1e-4, loss="mse"
    ):
        """
        num_iter: number of iterations
        max_depth: max depth of each tree
        nu: learning rate
        tol: tolerance for early stopping
        loss: loss function to use. DEFAULT: "mse"

        """
        self.num_iter = num_iter
        self.max_depth = max_depth
        self.nu = nu
        self.tol = tol
        self.loss = loss

    def fit(self, X, y):
        self.trees = []
        N, _ = X.shape
        self.y_mean = np.mean(y)
        fm = self.y_mean
        preds = [self.y_mean] * N
        for m in range(self.num_iter):
            r = y - preds  # residuals
            tree_i = DecisionTreeRegressor(
                max_depth=self.max_depth, random_state=0
            )
            tree_i.fit(X, r)
            self.trees.append(tree_i)
            if self.loss == "mse":
                preds = preds + self.nu * tree_i.predict(X)
            elif self.loss == "mae":
                preds = preds + self.nu * np.sign(tree_i.predict(X))
            if m > 0:
                if abs(preds - preds_old).sum() < self.tol:
                    break
            preds_old = preds

    def predict(self, X):
        y_hat = (
            np.array(
                list(map(lambda t: self.nu * t.predict(X), self.trees))
            ).sum(axis=0)
            + self.y_mean
        )
        return y_hat
