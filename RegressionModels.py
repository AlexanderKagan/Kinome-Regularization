import numpy as np
import cvxpy as cp
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter


class GroupLasso:
    def __init__(self, group_index, alpha=0.5, lambda1=1e-3):
        self.group_index = np.array(group_index)
        self.alpha = alpha
        self.lambda1 = lambda1

    def fit(self, X, y):
        assert X.shape[1] == len(self.group_index), "wrong number of variables"

        X_array = np.array(X)
        y_array = np.array(y, dtype=float)

        w = cp.Variable(X_array.shape[1])
        b = cp.Variable(1)

        group_2_num_elems = Counter(self.group_index)

        group_coef = list(map(lambda g: np.sqrt(group_2_num_elems[g]), self.group_index))
        regularizer = self.lambda1 * (self.alpha * cp.norm(w, 1) +
                                      (1 - self.alpha) * cp.scalar_product(group_coef, cp.square(w)))

        sum_squares = cp.sum_squares(b + w @ X_array.T - y_array)
        objective = cp.Minimize(sum_squares / (2 * len(y_array)) + regularizer)
        prob = cp.Problem(objective)
        _ = prob.solve()
        self.coef_ = w.value
        self.intercept_ = b.value

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class ClusterElasticNet:

    def __init__(self, n_clusters=50, lambda1=1e-3, lambda2=1e-3, max_iter=10, tol=1e-4, weight_update="full"):
        """
        Implements Cluster Elastic Net introduced in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4011669/
        :param n_clusters: int
        :param lambda1: float
        :param lambda2: float
        :param max_iter: int
        :param tol: float
        :param weight_update: str
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tol = tol
        self.max_iter = max_iter
        self.weight_update = weight_update
        self.n_clusters = n_clusters

    def _make_initial_estimation(self, X_array, y_array):
        n, p = X_array.shape
        bias = cp.Variable(1)
        beta = cp.Variable(p)

        objective = cp.Minimize(cp.sum_squares(X_array @ beta + bias - y_array) / (2 * n) +
                                self.lambda1 * cp.norm(beta, 1) + self.lambda2 * cp.norm(beta, 2))
        prob = cp.Problem(objective)
        prob.solve()

        self.coef_, self.intercept_ = beta.value, bias.value
        self.beta_history = [deepcopy(self.coef_)]
        self.labels_ = np.zeros_like(y_array)

    def fit(self, X, y, verbose=True):

        def make_weight_update():

            if self.weight_update == "full":
                for idx in range(p):
                    update_single_weight(idx)

            elif self.weight_update == "random":
                idx = np.random.choice(col_indices)
                update_single_weight(idx)
            else:
                raise NotImplementedError

        def update_single_weight(idx):

            def soft_threshold(a, b):
                return np.sign(a) * max(0., abs(a) - b)

            y_tilde = (y_array - self.intercept_) - np.delete(X_array, idx, 1) @ np.delete(self.coef_, idx)
            cluster = self.labels_[idx]
            cluster_mask = self.labels_ == cluster
            cluster_size = cluster_mask.sum()
            cluster_mask[idx] = False

            beta_idx = soft_threshold(X_array[:, idx] @ y_tilde +
                                      self.lambda2 / cluster_size * self.coef_[cluster_mask] @
                                      cov_matrix[idx][cluster_mask],
                                      self.lambda1 / 2) / \
                       cov_matrix[idx, idx] * (1. + self.lambda2 * (cluster_size - 1) / cluster_size)
            self.coef_[idx] = beta_idx

        X_array = np.array(X)
        y_array = np.array(y)
        n, p = X_array.shape

        cov_matrix = X_array.T @ X_array
        col_indices = np.arange(p)

        self._make_initial_estimation(X_array, y_array)

        for iteration in range(self.max_iter):
            # Step 1:
            kmeans = KMeans(n_clusters=self.n_clusters)
            X_kmeans = (X_array * np.vstack([self.coef_] * n)).T

            kmeans.fit(X_kmeans)
            self.labels_ = kmeans.labels_

            # Step 2:
            make_weight_update()

            # Verbose
            beta_change = np.linalg.norm(self.coef_ - self.beta_history[-1])

            if verbose:
                print(f"Iteration {iteration}, beta_change: {round(beta_change, 4)}")
            if beta_change < self.tol:
                break
            self.beta_history.append(deepcopy(self.coef_))

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
