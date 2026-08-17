from __future__ import annotations

import numpy as np
from hdbscan import HDBSCAN
from scipy.interpolate import interp1d
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


class DeconvolutionKDEPipeline:
    """
    Object-oriented pipeline for the deconvolution KDE preprocessing flow.

    This class keeps the stages modular:
    1. Build data-domain and frequency-domain grids.
    2. Compute the bandwidth and the tricube kernel Fourier transform.
    3. Compute empirical characteristic functions for observed data and noise.
    4. Deconvolve and smooth in the frequency domain.
    5. Invert to produce a density estimate in the data domain.
    6. Interpolate the density at the observed data points.
    7. Derive core distances from the density estimate.
    8. Build the mutual reachability matrix.
    9. Run HDBSCAN on the mutual reachability matrix.
    """

    def __init__(
        self,
        X_obs: np.ndarray,
        std: float,
        k: int,
        M: int = 512,
        buffer_frac: float = 0.1,
    ) -> None:
        """
        Initialize the pipeline with the given parameters.

        Parameters
        ----------
        X_obs : np.ndarray of shape (n_samples, n_features)
            The observed (noisy) dataset.
        std : float
            Known standard deviation of the Gaussian observation noise.
        k : int
            Number of nearest neighbours used for HDBSCAN min_samples and
            min_cluster_size.
        M : int, default 512
            Number of grid points for the density estimate. Best as a power
            of 2 for FFT efficiency.
        buffer_frac : float, default 0.1
            Fractional margin added around the data range to reduce edge
            effects on the density grid.
        """
        X_obs = np.asarray(X_obs, dtype=float)
        if X_obs.ndim != 2:
            raise ValueError("X_obs must be a 2D array of shape (n_samples, n_features).")
        if std < 0:
            raise ValueError("std must be non-negative.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k >= X_obs.shape[0]:
            raise ValueError("k must be smaller than the number of samples.")
        if M < 2:
            raise ValueError("M must be at least 2.")
        if buffer_frac < 0:
            raise ValueError("buffer_frac must be non-negative.")

        self.X_obs = X_obs
        self.std = std
        self.k = k
        self.M = M
        self.buffer_frac = buffer_frac
        self.n = X_obs.shape[0]

        # Fitted results stored after run()
        self.x_grid = None
        self.t_grid = None
        self.separation = None
        self.h = None
        self.phi_K = None
        self.phi_X = None
        self.phi_U = None
        self.phi_U_safe = None
        self.f_hat = None
        self.density_at_points = None
        self.core_distances = None
        self.mutual_reachability_matrix = None
        self.predicted_labels = None

    @staticmethod
    def _pairwise_distances(X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    def build_grids(
        self,
        X_obs: np.ndarray,
        M: int,
        buffer_frac: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Build the evenly spaced data-domain grid and the corresponding
        frequency-domain grid.

        Returns
        -------
        x_grid : np.ndarray of shape (M,)
            Evenly spaced points spanning the range of X_obs with added buffer.
        t_grid : np.ndarray of shape (M,)
            Corresponding angular frequency grid (radians per unit).
        separation : float
            Distance between consecutive x_grid points.
        """
        buffer = buffer_frac * (X_obs.max() - X_obs.min())

        x_min = X_obs.min() - buffer
        x_max = X_obs.max() + buffer

        x_grid = np.linspace(x_min, x_max, M)
        separation = x_grid[1] - x_grid[0]
        t_grid = 2 * np.pi * np.fft.fftfreq(M, d=separation)

        return x_grid, t_grid, separation

    def compute_bandwidth(self, X_obs: np.ndarray) -> float:
        """
        Compute the global smoothing bandwidth h.

        Uses a noise-integrated modification of the Gaussian AMISE formula so
        that h grows with noise and shrinks with sample size.

        Returns
        -------
        h : float
            Bandwidth value.
        """
        n = X_obs.shape[0]
        h = self.std * (np.log(n) / n) ** (1 / 5)
        return h

    @staticmethod
    def compute_kernel_ft(t_grid: np.ndarray, h: float) -> np.ndarray:
        """
        Evaluate the Fourier transform of the tricube kernel at scaled
        frequency grid points.

        The tricube kernel has compact support on [-1, 1] and is set to zero
        outside that interval, acting as a low-pass filter.

        Parameters
        ----------
        t_grid : np.ndarray of shape (M,)
            Angular frequency grid.
        h : float
            Bandwidth used to scale the frequency grid before applying the
            kernel.

        Returns
        -------
        phi_K : np.ndarray of shape (M,)
            Kernel Fourier transform values.
        """
        u = t_grid * h
        inner = (1 - np.abs(u) ** 3) ** 3
        inner[np.abs(u) > 1] = 0.0
        return (70 / 81) * inner

    @staticmethod
    def compute_characteristic_functions(
        X_obs: np.ndarray,
        t_grid: np.ndarray,
        std: float,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the empirical characteristic function of the observed data
        and the analytic characteristic function of the Gaussian noise.

        Parameters
        ----------
        X_obs : np.ndarray of shape (n_samples, n_features)
            Observed data.
        t_grid : np.ndarray of shape (M,)
            Angular frequency grid.
        std : float
            Known Gaussian noise standard deviation.
        eps : float, default 1e-6
            Floor value applied to phi_U to prevent division by zero.

        Returns
        -------
        phi_X : np.ndarray of shape (M,)
            Empirical characteristic function of the observed data.
        phi_U : np.ndarray of shape (M,)
            Analytic characteristic function of the noise.
        phi_U_safe : np.ndarray of shape (M,)
            phi_U with very small values replaced by eps.
        """
        phi_X = np.mean(np.exp(1j * np.outer(t_grid, X_obs)), axis=1)
        phi_U = np.exp(-0.5 * (std ** 2) * (t_grid ** 2))
        phi_U_safe = np.where(np.abs(phi_U) < eps, eps, phi_U)

        return phi_X, phi_U, phi_U_safe

    @staticmethod
    def deconvolve_and_smooth(
        phi_X: np.ndarray,
        phi_U_safe: np.ndarray,
        phi_K: np.ndarray,
    ) -> np.ndarray:
        """
        Deconvolve the observed data characteristic function by dividing out
        the noise, then apply the kernel low-pass filter.

        Parameters
        ----------
        phi_X : np.ndarray of shape (M,)
        phi_U_safe : np.ndarray of shape (M,)
        phi_K : np.ndarray of shape (M,)

        Returns
        -------
        smoothed_phi_Y : np.ndarray of shape (M,)
            Deconvolved and smoothed characteristic function of the true data.
        """
        phi_Y = phi_X / phi_U_safe
        return phi_Y * phi_K

    @staticmethod
    def estimate_density(
        smoothed_phi_Y: np.ndarray,
        x_grid: np.ndarray,
        M: int,
    ) -> np.ndarray:
        """
        Invert the smoothed characteristic function to recover the density
        estimate in the data domain, then normalize it to integrate to 1.

        Parameters
        ----------
        smoothed_phi_Y : np.ndarray of shape (M,)
        x_grid : np.ndarray of shape (M,)
        M : int
            Number of grid points.

        Returns
        -------
        f_hat : np.ndarray of shape (M,)
            Normalized density estimate at x_grid points.
        """
        f_hat = np.real(np.fft.ifft(np.fft.ifftshift(smoothed_phi_Y))) * M / (2 * np.pi)
        f_hat /= np.trapz(f_hat, x_grid)
        return f_hat

    @staticmethod
    def interpolate_density(
        f_hat: np.ndarray,
        x_grid: np.ndarray,
        X_obs: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate the grid-based density estimate at the first feature
        coordinate of each observed point.

        Parameters
        ----------
        f_hat : np.ndarray of shape (M,)
            Density estimate on x_grid.
        x_grid : np.ndarray of shape (M,)
            Grid points.
        X_obs : np.ndarray of shape (n_samples, n_features)
            Observed data points.

        Returns
        -------
        density_at_points : np.ndarray of shape (n_samples,)
            Interpolated density value at each observed point.
        """
        interp_func = interp1d(x_grid, f_hat, kind="linear", fill_value="extrapolate")
        return interp_func(X_obs[:, 0])

    @staticmethod
    def calculate_core_distances(
        density_at_points: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Derive core distances from the density estimate by taking the
        inverse of the density.

        Points in dense regions get small core distances; sparse points get
        large core distances.

        Parameters
        ----------
        density_at_points : np.ndarray of shape (n_samples,)
        eps : float, default 1e-8
            Small floor added to the density to prevent division by zero.

        Returns
        -------
        core_distances : np.ndarray of shape (n_samples,)
        """
        return 1.0 / (density_at_points + eps)

    @staticmethod
    def mutual_reachability(
        core_distances: np.ndarray,
        X_obs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the mutual reachability distance matrix.

        For each pair (i, j) the mutual reachability distance is:
            max(core_dist(i), core_dist(j), d(i, j))

        Parameters
        ----------
        core_distances : np.ndarray of shape (n_samples,)
        X_obs : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        mutual : np.ndarray of shape (n_samples, n_samples)
        """
        n_samples = X_obs.shape[0]
        if core_distances.shape != (n_samples,):
            raise ValueError("core_distances must have shape (n_samples,).")

        observed_distances = np.linalg.norm(
            X_obs[:, None, :] - X_obs[None, :, :], axis=2
        )

        core_distances_i = core_distances[:, None]
        core_distances_j = core_distances[None, :]

        return np.maximum(
            np.maximum(core_distances_i, core_distances_j),
            observed_distances,
        )

    def run_fast_hdbscan(self, mutual: np.ndarray, k: int) -> np.ndarray:
        """
        Run HDBSCAN on the mutual reachability matrix to produce cluster labels.

        Parameters
        ----------
        mutual : np.ndarray of shape (n_samples, n_samples)
            Precomputed mutual reachability distance matrix.
        k : int
            Used for both min_samples and min_cluster_size.

        Returns
        -------
        pred_labels : np.ndarray of shape (n_samples,)
            Cluster label per point. Noise points are labelled -1.
        """
        clusterer = HDBSCAN(
            metric="precomputed",
            min_samples=k,
            min_cluster_size=k,
        )

        return clusterer.fit_predict(mutual)

    @staticmethod
    def calculate_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Compute the Adjusted Rand Index between true and predicted labels.

        Parameters
        ----------
        true_labels : np.ndarray of shape (n_samples,)
        pred_labels : np.ndarray of shape (n_samples,)

        Returns
        -------
        ari : float in [-1, 1]
        """
        return adjusted_rand_score(true_labels, pred_labels)

    def run(self) -> np.ndarray:
        """
        Execute the full deconvolution KDE pipeline and return predicted labels.

        Intermediate results are stored as instance attributes after this call:
        x_grid, t_grid, separation, h, phi_K, phi_X, phi_U, phi_U_safe,
        f_hat, density_at_points, core_distances, mutual_reachability_matrix,
        predicted_labels.

        Returns
        -------
        predicted_labels : np.ndarray of shape (n_samples,)
        """
        x_grid, t_grid, separation = self.build_grids(
            self.X_obs, self.M, self.buffer_frac
        )
        h = self.compute_bandwidth(self.X_obs)
        phi_K = self.compute_kernel_ft(t_grid, h)
        phi_X, phi_U, phi_U_safe = self.compute_characteristic_functions(
            self.X_obs, t_grid, self.std
        )
        smoothed_phi_Y = self.deconvolve_and_smooth(phi_X, phi_U_safe, phi_K)
        f_hat = self.estimate_density(smoothed_phi_Y, x_grid, self.M)
        density_at_points = self.interpolate_density(f_hat, x_grid, self.X_obs)
        core_distances = self.calculate_core_distances(density_at_points)
        mutual = self.mutual_reachability(core_distances, self.X_obs)
        pred_labels = self.run_fast_hdbscan(mutual, self.k)

        self.x_grid = x_grid
        self.t_grid = t_grid
        self.separation = separation
        self.h = h
        self.phi_K = phi_K
        self.phi_X = phi_X
        self.phi_U = phi_U
        self.phi_U_safe = phi_U_safe
        self.f_hat = f_hat
        self.density_at_points = density_at_points
        self.core_distances = core_distances
        self.mutual_reachability_matrix = mutual
        self.predicted_labels = pred_labels

        return pred_labels

    def main(self) -> np.ndarray:
        """Run the full pipeline. Alias for run()."""
        return self.run()
