from __future__ import annotations

import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

class ProbabilityRankPipeline:
    """
    Object-oriented pipeline for the regular probability-rank preprocessing flow.

    This class keeps the three requested steps modular:
    1. build the regular probability-rank matrix with Monte Carlo simulations,
    2. allocate uncertainty-aware core distances,
    3. run fast_hdbscan clustering on the induced mutual-reachability matrix.
    """

    def __init__(self, std: float, dataset: np.ndarray, sims: int, k: int) -> None:
        """Initialize the pipeline with the given parameters."""
        dataset = np.asarray(dataset, dtype=float)
        if dataset.ndim != 2:
            raise ValueError("dataset must be a 2D array.")
        if std < 0:
            raise ValueError("std must be non-negative.")
        if sims <= 0:
            raise ValueError("sims must be a positive integer.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k >= dataset.shape[0]:
            raise ValueError("k must be smaller than the number of samples.")

        self.std = std
        self.dataset = dataset
        self.n = self.dataset.shape[0]
        self.sims = sims
        self.k = k
        self.rng = np.random.default_rng()

        self.fixed_observed_dataset = self._sample_observed_dataset(
            self.dataset,
            self.std,
        )
        self.probability_rank_matrix = None
        self.core_distances = None
        self.mutual_reachability_matrix = None
        self.predicted_labels = None

    @staticmethod
    def _pairwise_distances(dataset: np.ndarray) -> np.ndarray:
        """Computes the pairwise euclidian distances between rows of the dataset."""
        return np.linalg.norm(dataset[:, None, :] - dataset[None, :, :], axis=2)

    def _sample_observed_dataset(
        self,
        dataset: np.ndarray,
        std: float,
    ) -> np.ndarray:
        """Samples a new observed dataset by adding Gaussian noise to the original dataset."""
        return dataset + self.rng.normal(scale=std, size=dataset.shape)

    def create_base_matrix(self, n: int) -> np.ndarray:
        """Creates an empty base matrix to accumulate counts of close-enough pairs."""
        return np.zeros((n, n), dtype=np.int32)

    def create_probability_rank_matrix(
        self,
        base_matrix: np.ndarray,
        std: float,
        dataset: np.ndarray,
        sims: int,
        k: int,
    ) -> np.ndarray:
        """Creates the probabaility rank matric by running Monte Carlo simulations and counting close-enough pairs."""
        n_samples = dataset.shape[0]

        for _ in range(sims):
            observed_dataset = self._sample_observed_dataset(dataset, std)
            euclidean_distances = self._pairwise_distances(observed_dataset)

            for i in range(n_samples):
                for j in range(n_samples):
                    if i == j:
                        continue
                    num_close_enough = (
                        euclidean_distances[i] < euclidean_distances[i, j]
                    ).sum()
                    if num_close_enough - 1 >= k:
                        base_matrix[i, j] += 1

        return base_matrix / sims

    def calculate_core_distances(
        self,
        probability_rank_matrix: np.ndarray,
        observed_dataset_fixed: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Calculates the uncertainty-aware core distances between each point and its k-th nearest neighbour in the observed dataset, using the probability rank matrix to determine which points are close enough."""
        n_samples = observed_dataset_fixed.shape[0]
        if probability_rank_matrix.shape != (n_samples, n_samples):
            raise ValueError(
                "probability_rank_matrix must have shape (n_samples, n_samples)."
            )

        nbrs = NearestNeighbors(n_neighbors=n_samples).fit(observed_dataset_fixed)
        distances, neighbors = nbrs.kneighbors(observed_dataset_fixed)

        core_distances = np.zeros(n_samples, dtype=float)

        for i in range(n_samples):
            total = 0.0
            for idx in range(distances.shape[1]):
                j = neighbors[i, idx]
                if j == i:
                    continue

                total += probability_rank_matrix[i, j]
                if total >= k:
                    core_distances[i] = distances[i, idx]
                    break
            else:
                core_distances[i] = distances[i, -1]

        return core_distances

    def mutual_reachability(
        self,
        core_distances: np.ndarray,
        observed_dataset_fixed: np.ndarray,
    ) -> np.ndarray:
        """Calculates the mutual reachibility matrix using the core distances and the observed dataset."""
        n_samples = observed_dataset_fixed.shape[0]
        if core_distances.shape != (n_samples,):
            raise ValueError("core_distances must have shape (n_samples,).")

        observed_distances = self._pairwise_distances(observed_dataset_fixed)

        core_distances_i = core_distances[:, None]
        core_distances_j = core_distances[None, :]
        if core_distances_i.shape != (n_samples, 1):
            raise ValueError("core_distances_i must have shape (n_samples, 1).")
        if core_distances_j.shape != (1, n_samples):
            raise ValueError("core_distances_j must have shape (1, n_samples).")

        return np.maximum(
            np.maximum(core_distances_i, core_distances_j),
            observed_distances,
        )

    def run_fast_hdbscan(self, mutual: np.ndarray, k: int) -> np.ndarray:
        """Runs fast_hdbscan on the mutual reachability matrix to obtain cluster labels."""
        clusterer = HDBSCAN(
            metric="precomputed",
            min_samples=k,
            min_cluster_size=k,
        )
        
        return clusterer.fit_predict(mutual)

    @staticmethod
    def calculate_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """Calculates the Adjusted Rand Index between the true labels and the predicted labels."""
        return adjusted_rand_score(true_labels, pred_labels)

    def run(self) -> np.ndarray:
        """Runs the full probability rank pipeline and returns the predicted cluster labels."""
        base_matrix = self.create_base_matrix(self.n)
        probability_rank_matrix = self.create_probability_rank_matrix(
            base_matrix,
            self.std,
            self.dataset,
            self.sims,
            self.k,
        )
        core_distances = self.calculate_core_distances(
            probability_rank_matrix,
            self.fixed_observed_dataset,
            self.k,
        )
        mutual = self.mutual_reachability(
            core_distances,
            self.fixed_observed_dataset,
        )
        pred_labels = self.run_fast_hdbscan(mutual, self.k)

        self.probability_rank_matrix = probability_rank_matrix
        self.core_distances = core_distances
        self.mutual_reachability_matrix = mutual
        self.predicted_labels = pred_labels

        return pred_labels

    def main(self) -> np.ndarray:
        """Main method to run the pipeline and return predicted labels."""
        return self.run()