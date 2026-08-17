# Copilot instructions for New-HDBSCAN

## Big picture
- This repository has two distinct code areas:
  - `fast_hdbscan/`: the reusable clustering library. Treat this as the stable algorithm implementation.
  - `new_hdbscan/`: experimental research code that layers preprocessing ideas on top of HDBSCAN.
- `trials/` contains notebooks used to prototype ideas before moving them into `new_hdbscan/` modules.
- The top-level `README.md` is a research to-do log, not the authoritative source for build or test workflows.

## Core architecture
- The main library entry points are in `fast_hdbscan/fast_hdbscan/__init__.py`: `HDBSCAN`, `LayerClustering`, `fast_hdbscan`, `BranchDetector`, and `find_branch_sub_clusters`.
- The clustering pipeline in `fast_hdbscan/fast_hdbscan/hdbscan.py` is:
  1. validate feature-vector input,
  2. build a `KDTree`,
  3. compute a Borůvka minimum spanning tree,
  4. derive linkage / condensed trees,
  5. extract clusters and probabilities.
- `fast_hdbscan.HDBSCAN` is designed for low-dimensional Euclidean feature data, not precomputed distance matrices.
- Experimental preprocessing that produces a precomputed mutual-reachability matrix belongs in `new_hdbscan/probability_rank_matrix/`, where the current object-oriented entry point is `ProbabilityRankPipeline`.

## Project-specific patterns
- Keep changes to `fast_hdbscan/` minimal and library-like: sklearn-style estimators, NumPy arrays, validation early, and explicit return values.
- `fast_hdbscan/fast_hdbscan/__init__.py` intentionally triggers JIT compilation on import by fitting small random datasets; avoid changing this casually.
- Tests in `fast_hdbscan/fast_hdbscan/tests/` use direct NumPy assertions and pytest functions rather than heavy fixtures.
- Handle non-finite data the same way as the library does: assign such points to noise (`-1`) rather than silently dropping them.
- Put new research code in `new_hdbscan/` instead of modifying the vendored library unless the algorithm itself truly changes.
- Do not edit `env/`; it is a local virtual environment, not project source.

## Workflows that matter
- Install library dependencies from the nested package:
  - `cd fast_hdbscan`
  - `pip install -r requirements.txt`
  - `pip install -e .`
- Main test command from CI:
  - `pytest fast_hdbscan/tests --show-capture=no -v --disable-warnings`
- CI also uses coverage:
  - `pytest fast_hdbscan/tests --cov=fast_hdbscan/ --cov-report=xml --cov-report=html`
- Packaging for the library is driven by `fast_hdbscan/setup.cfg` and `fast_hdbscan/pyproject.toml`; supported Python versions are 3.9-3.12.

## Guidance for AI coding agents
- When implementing an idea from a notebook, first look for the nearest modular home in `new_hdbscan/`, then keep a thin API that notebooks can call.
- If you need to expose a new stable feature, mirror existing top-level exports from `fast_hdbscan/fast_hdbscan/__init__.py`.
- Use `fast_hdbscan/README.rst`, `fast_hdbscan/azure-pipelines.yml`, and `fast_hdbscan/fast_hdbscan/tests/test_hdbscan.py` as the primary examples of intended library usage and validation.
- For probability-rank work, prefer the class-based API in `new_hdbscan/probability_rank_matrix/probability_rank.py` and keep the regular probability-rank variant separate from notebook-only experiments.
