"""Residual Quantization with Stratified Plasticity.

Core library for the paper "Stable Semantic IDs under Distribution Shift".
Pure numpy/scipy implementation — no external dependencies beyond these.

Classes:
    RQCodebook: Residual quantizer with greedy stage-wise k-means.
                Supports uniform and funnel (non-uniform K) architectures.

Functions:
    warm_retrain: Warm-retrain suffix stages on shifted data.
    gap_recovery: Compute the gap recovery ratio rho.
    codebook_entropy: Shannon entropy of codebook usage per stage.
    retrieval_recall_at_k: Recall@K via asymmetric decode.
    generate_data: Gaussian blob data for synthetic experiments.
    apply_drift: Apply mean-shift, scale, or rotation drift.
    code_consistency: Fraction of identical code sequences.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def _kmeans(
    X: np.ndarray,
    k: int,
    n_iter: int = 20,
    rng: np.random.RandomState | None = None,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """K-means clustering with k-means++ init. Returns centroids (k, dim)."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(X)

    if init is not None:
        centroids = init.copy()
    else:
        centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
        centroids[0] = X[rng.randint(n)]
        for i in range(1, k):
            dists = np.min(
                np.sum((X[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
                axis=1,
            )
            total = dists.sum()
            if total < 1e-12:
                centroids[i] = X[rng.randint(n)]
            else:
                probs = dists / total
                centroids[i] = X[rng.choice(n, p=probs)]

    for _ in range(n_iter):
        assignments = _assign(X, centroids)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                centroids[j] = X[mask].mean(axis=0)

    return centroids


def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each point to nearest centroid. Returns (n,) int array."""
    dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1).astype(np.int64)


class RQCodebook:
    """Residual Quantizer with support for non-uniform codebook sizes (funnel).

    Args:
        n_stages: Number of quantization stages M.
        codes_per_stage: Either a single int (uniform) or a list of ints
            (one per stage). For the funnel architecture, use e.g.
            [16, 16, 256, 512].
        dim: Embedding dimensionality.
    """

    def __init__(
        self,
        n_stages: int,
        codes_per_stage: int | Sequence[int],
        dim: int,
    ):
        self.n_stages = n_stages
        self.dim = dim
        if isinstance(codes_per_stage, int):
            self.codes_per_stage = [codes_per_stage] * n_stages
        else:
            assert len(codes_per_stage) == n_stages
            self.codes_per_stage = list(codes_per_stage)
        self.codebooks: list[np.ndarray] = []

    @property
    def bitrate(self) -> float:
        return sum(np.log2(k) for k in self.codes_per_stage)

    def fit(
        self, X: np.ndarray, n_iter: int = 20, seed: int = 42
    ) -> RQCodebook:
        """Train codebooks greedily, stage by stage."""
        rng = np.random.RandomState(seed)
        residual = X.copy()
        self.codebooks = []

        for m in range(self.n_stages):
            k = self.codes_per_stage[m]
            centroids = _kmeans(residual, k, n_iter=n_iter, rng=rng)
            self.codebooks.append(centroids)
            assignments = _assign(residual, centroids)
            residual = residual - centroids[assignments]

        return self

    def encode(
        self, X: np.ndarray, n_stages: int | None = None
    ) -> list[np.ndarray]:
        """Encode X through the RQ stages. Returns list of assignment arrays."""
        if n_stages is None:
            n_stages = len(self.codebooks)
        residual = X.copy()
        codes = []
        for m in range(n_stages):
            assignments = _assign(residual, self.codebooks[m])
            codes.append(assignments)
            residual = residual - self.codebooks[m][assignments]
        return codes

    def decode(self, codes: list[np.ndarray]) -> np.ndarray:
        """Reconstruct from code sequences."""
        X_hat = np.zeros((len(codes[0]), self.dim), dtype=np.float32)
        for m, c in enumerate(codes):
            X_hat += self.codebooks[m][c]
        return X_hat

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode then decode."""
        return self.decode(self.encode(X))

    def mse(self, X: np.ndarray) -> float:
        """Reconstruction MSE on X."""
        X_hat = self.reconstruct(X)
        return float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))

    def get_residual(
        self, X: np.ndarray, n_stages: int | None = None
    ) -> np.ndarray:
        """Get the residual after n_stages of quantization."""
        if n_stages is None:
            n_stages = len(self.codebooks)
        residual = X.copy()
        for m in range(n_stages):
            assignments = _assign(residual, self.codebooks[m])
            residual = residual - self.codebooks[m][assignments]
        return residual

    def per_stage_mse(self, X: np.ndarray) -> list[float]:
        """Cumulative MSE after each stage."""
        errors = []
        residual = X.copy()
        reconstruction = np.zeros_like(X)
        for m in range(len(self.codebooks)):
            assignments = _assign(residual, self.codebooks[m])
            reconstruction += self.codebooks[m][assignments]
            residual = X - reconstruction
            errors.append(float(np.mean(np.sum(residual ** 2, axis=1))))
        return errors

    def codebook_utilization(self, X: np.ndarray) -> list[float]:
        """Fraction of codebook entries used at each stage."""
        utilizations = []
        residual = X.copy()
        for m in range(len(self.codebooks)):
            assignments = _assign(residual, self.codebooks[m])
            n_used = len(np.unique(assignments))
            utilizations.append(n_used / self.codes_per_stage[m])
            residual = residual - self.codebooks[m][assignments]
        return utilizations


def warm_retrain(
    rq: RQCodebook,
    X_new: np.ndarray,
    freeze_depth: int | None = None,
    n_iter: int = 20,
    seed: int = 42,
    spectral: bool = False,
    rq_source: RQCodebook | None = None,
) -> RQCodebook:
    """Warm-retrain suffix stages on shifted data.

    Freezes stages 1..freeze_depth, warm-retrains the rest using old
    centroids as initialization (default) or spectral initialization.

    Args:
        rq: Trained RQ codebook (not modified).
        X_new: Data from the shifted distribution.
        freeze_depth: Number of prefix stages to freeze.
            Default: floor(M/2).
        n_iter: K-means iterations for retraining.
        seed: Random seed.
        spectral: If True, initialize the first suffix stage from the
            leading principal components of the drift residual instead
            of warm-starting from old centroids.
        rq_source: Source-period codebook for computing drift residuals.
            Required when spectral=True. If None, uses rq.

    Returns:
        New RQCodebook with frozen prefix and retrained suffix.
    """
    if freeze_depth is None:
        freeze_depth = len(rq.codebooks) // 2

    rq_new = RQCodebook(rq.n_stages, rq.codes_per_stage, rq.dim)
    rq_new.codebooks = [cb.copy() for cb in rq.codebooks]

    residual = X_new.copy()
    for m in range(freeze_depth):
        assignments = _assign(residual, rq_new.codebooks[m])
        residual = residual - rq_new.codebooks[m][assignments]

    rng = np.random.RandomState(seed)

    if spectral and freeze_depth < rq.n_stages:
        src = rq_source if rq_source is not None else rq
        source_residual = src.get_residual(X_new, n_stages=freeze_depth)
        drift = residual - source_residual
        K_first = rq.codes_per_stage[freeze_depth]
        mean_res = residual.mean(axis=0)
        centered = residual - mean_res
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        explained = np.cumsum(S**2) / max(np.sum(S**2), 1e-12)
        r = int(np.searchsorted(explained, 0.9)) + 1
        r = min(r, Vt.shape[0])
        projections = centered @ Vt[:r].T
        spectral_init = np.zeros(
            (K_first, residual.shape[1]), dtype=np.float32
        )
        for j in range(min(r, K_first)):
            lo = np.percentile(projections[:, j % r], 100 * j / K_first)
            hi = np.percentile(projections[:, j % r], 100 * (j + 1) / K_first)
            mask = (projections[:, j % r] >= lo) & (projections[:, j % r] < hi)
            if mask.sum() > 0:
                spectral_init[j] = residual[mask].mean(axis=0)
            else:
                spectral_init[j] = residual[rng.randint(len(residual))]
        for j in range(min(r, K_first), K_first):
            spectral_init[j] = residual[rng.randint(len(residual))]
        centroids = _kmeans(
            residual, K_first, n_iter=n_iter, rng=rng,
            init=spectral_init,
        )
        rq_new.codebooks[freeze_depth] = centroids
        assignments = _assign(residual, centroids)
        residual = residual - centroids[assignments]
        start = freeze_depth + 1
    else:
        start = freeze_depth

    for m in range(start, rq.n_stages):
        centroids = _kmeans(
            residual,
            rq.codes_per_stage[m],
            n_iter=n_iter,
            rng=rng,
            init=rq_new.codebooks[m],
        )
        rq_new.codebooks[m] = centroids
        assignments = _assign(residual, centroids)
        residual = residual - centroids[assignments]

    return rq_new


def gap_recovery(mse_frozen: float, mse_warm: float, mse_full: float) -> float:
    """Compute gap recovery ratio rho (Eq. 1 in the paper)."""
    denom = mse_frozen - mse_full
    if abs(denom) < 1e-12:
        return 1.0
    return 1.0 - (mse_warm - mse_full) / denom


def codebook_entropy(rq: RQCodebook, X: np.ndarray) -> list[float]:
    """Shannon entropy of codebook usage at each stage (in bits)."""
    entropies = []
    residual = X.copy()
    for m in range(len(rq.codebooks)):
        assignments = _assign(residual, rq.codebooks[m])
        counts = np.bincount(assignments, minlength=rq.codes_per_stage[m])
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies.append(float(-np.sum(probs * np.log2(probs))))
        residual = residual - rq.codebooks[m][assignments]
    return entropies


def retrieval_recall_at_k(
    queries: np.ndarray,
    database: np.ndarray,
    rq_query: RQCodebook,
    rq_db: RQCodebook,
    k: int = 10,
    n_queries: int = 1000,
    n_db_sample: int = 5000,
    seed: int = 42,
) -> float:
    """Recall@K using asymmetric decode.

    Query and database may use different suffix codebooks.
    Ground truth is brute-force L2 on raw vectors.
    """
    rng = np.random.RandomState(seed)
    q_idx = rng.choice(
        len(queries), min(n_queries, len(queries)), replace=False
    )
    db_idx = rng.choice(
        len(database), min(n_db_sample, len(database)), replace=False
    )

    q_raw = queries[q_idx]
    db_raw = database[db_idx]

    dists_true = np.sum(
        (q_raw[:, None, :] - db_raw[None, :, :]) ** 2, axis=2
    )
    true_nn = np.argsort(dists_true, axis=1)[:, :k]

    q_decoded = rq_query.decode(rq_query.encode(q_raw))
    db_decoded = rq_db.decode(rq_db.encode(db_raw))

    dists_approx = np.sum(
        (q_decoded[:, None, :] - db_decoded[None, :, :]) ** 2, axis=2
    )
    approx_nn = np.argsort(dists_approx, axis=1)[:, :k]

    recall = 0.0
    for i in range(len(q_idx)):
        true_set = set(true_nn[i].tolist())
        approx_set = set(approx_nn[i].tolist())
        recall += len(true_set & approx_set) / k
    return recall / len(q_idx)


def code_consistency(
    codes_a: list[np.ndarray], codes_b: list[np.ndarray]
) -> float:
    """Fraction of code sequences that are identical across two encodings."""
    n = len(codes_a[0])
    match = np.ones(n, dtype=bool)
    for m in range(len(codes_a)):
        match &= codes_a[m] == codes_b[m]
    return float(match.mean())


def prefix_consistency(
    codes_a: list[np.ndarray],
    codes_b: list[np.ndarray],
    freeze_depth: int,
) -> float:
    """Fraction of prefix codes that are identical."""
    n = len(codes_a[0])
    match = np.ones(n, dtype=bool)
    for m in range(freeze_depth):
        match &= codes_a[m] == codes_b[m]
    return float(match.mean())


# ---------------------------------------------------------------------------
# Data generation utilities
# ---------------------------------------------------------------------------


def generate_data(
    n_samples: int,
    dim: int,
    n_clusters: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Generate Gaussian blob data."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32) * 3.0
    labels = rng.randint(0, n_clusters, size=n_samples)
    X = centers[labels] + rng.randn(n_samples, dim).astype(np.float32) * 0.5
    return X


def apply_drift(
    X: np.ndarray,
    drift_type: str = "mean_shift",
    magnitude: float = 1.0,
    seed: int = 123,
) -> np.ndarray:
    """Apply distribution drift to data.

    Args:
        X: Input data (n, d).
        drift_type: One of "mean_shift", "scale", "rotation".
        magnitude: Drift strength.
        seed: Random seed for drift direction.

    Returns:
        Shifted data (n, d).
    """
    rng = np.random.RandomState(seed)

    if drift_type == "mean_shift":
        direction = rng.randn(X.shape[1]).astype(np.float32)
        direction /= np.linalg.norm(direction)
        return X + direction * magnitude

    elif drift_type == "scale":
        return X * (1.0 + magnitude * 0.5)

    elif drift_type == "rotation":
        from scipy.stats import special_ortho_group

        R = special_ortho_group.rvs(X.shape[1], random_state=rng).astype(
            np.float32
        )
        t = min(magnitude / 5.0, 1.0)
        R_interp = (1 - t) * np.eye(X.shape[1], dtype=np.float32) + t * R
        U, _, Vt = np.linalg.svd(R_interp)
        R_interp = (U @ Vt).astype(np.float32)
        return X @ R_interp.T

    else:
        raise ValueError(f"Unknown drift type: {drift_type}")
