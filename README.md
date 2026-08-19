# New-HDBSCAN: Density Estimation and Deconvolution Extensions

Research into improving HDBSCAN's density estimation and probability calculations through deconvolution-based techniques for robust clustering in noisy, non-convex, and high-dimensional data.

## Overview

This project extends HDBSCAN with novel density estimation methods to improve cluster stability and probability estimates, particularly in the presence of noise. The core contributions are a **monte carlo precomputation** and a **deconvolution-based pipeline** that separates true cluster structure from noise corruption, enabling more accurate density and persistence calculations.

## Project Status

### Completed
- Monte Carlo pre-computation
- Initial deconvolution prototype (Step 0 & Step 1)
- Adaptive h-value construction for noise handling

### In Progress
- Implementing change-of-variables formula for density

### Planned
- Extend noise types beyond Gaussian (Laplacian, mixture distributions)
- Benchmark on real-world datasets

## Test Suite

Six benchmark scenarios covering real-world challenges:

1. **Non-Convex Shapes** — half-moons, concentric circles, spirals
2. **Varying Densities** — multi-scale Gaussian blobs, background clutter
3. **Anisotropic Clusters** — elongated, linearly-transformed blobs
4. **Hierarchical (Nested) Clusters** — Gaussian-in-Gaussian-in-Gaussian
5. **High-Dimensional Manifolds** — Swiss-roll in 10+ dims
6. **Heavy-Tailed Distributions** — Gaussian core + Cauchy/t-distributed outliers

## Dependencies

- Python 3.10+
- `hdbscan` — baseline clustering
- `numpy`, `scipy` — numerical operations
- `scikit-learn` — synthetic data and metrics
- `matplotlib` — visualization

## License
