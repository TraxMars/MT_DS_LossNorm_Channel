"""Descriptive statistics in shared parameter coordinates, with explicit masks."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from core import gradient_validity, signed_cosine


def masked_mean(values, axis=None):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def describe(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return dict(n=0, mean=None, median=None, q10=None, q90=None, positive_fraction=None)
    return dict(n=len(array), mean=float(array.mean()), median=float(np.median(array)),
                q10=float(np.quantile(array, .1)), q90=float(np.quantile(array, .9)),
                positive_fraction=float(np.mean(array > 0)))


def pairwise_cosine(left, right, absolute_floor=1e-12, relative_floor=1e-6):
    """[cases,channels,kernel] -> [channels,left cases,right cases]."""
    ln, ls, _ = gradient_validity(left, absolute_floor, relative_floor)
    rn, rs, _ = gradient_validity(right, absolute_floor, relative_floor)
    valid = ((ls == "valid").T[:, :, None] & (rs == "valid").T[:, None, :])
    dot = np.matmul(np.asarray(left, dtype=np.float64).transpose(1, 0, 2),
                    np.asarray(right, dtype=np.float64).transpose(1, 2, 0))
    scores = np.divide(dot, ln.T[:, :, None] * rn.T[:, None, :],
                       out=np.full(dot.shape, np.nan), where=valid)
    return np.clip(scores, -1, 1), valid, ls, rs


def bootstrap_pair_mean(scores, seed=142, draws=200):
    """Resample L and U cases separately; never treat Cartesian pairs as iid."""
    channels, n_left, n_right = scores.shape
    generator = np.random.default_rng(seed)
    finite = np.isfinite(scores)
    values = np.where(finite, scores, 0)
    means = np.full((draws, channels), np.nan)
    for draw in range(draws):
        left_weights = np.bincount(generator.integers(0, n_left, n_left), minlength=n_left)
        right_weights = np.bincount(generator.integers(0, n_right, n_right), minlength=n_right)
        weights = np.outer(left_weights, right_weights)
        denominator = (finite * weights).sum(axis=(1, 2))
        means[draw] = np.divide((values * weights).sum(axis=(1, 2)), denominator,
                                out=np.full(channels, np.nan), where=denominator > 0)
    bounds = np.full((2, channels), np.nan)
    for channel in range(channels):
        valid = means[:, channel][np.isfinite(means[:, channel])]
        if len(valid):
            bounds[:, channel] = np.quantile(valid, [.025, .975])
    return bounds


def case_means(array, repeats, factors=None):
    count = array.shape[0] // repeats
    if array.shape[0] % repeats:
        raise ValueError("Observation count is not divisible by repeat count")
    means = np.empty((count, *array.shape[1:]), dtype=np.float64)
    for case in range(count):
        block = np.asarray(array[case*repeats:(case+1)*repeats], dtype=np.float64)
        if factors is not None:
            block = block * np.asarray(factors[case*repeats:(case+1)*repeats])[:, None, None, None]
        means[case] = block.mean(axis=0)
    return means


def observation_patterns(array, reachable_indices, absolute_floor, relative_floor):
    """Within-patch cross-head directions, SVD and cancellation; no space matching."""
    n_obs, _, channels, _ = array.shape
    reachable = list(reachable_indices)
    fractions = np.full((n_obs, channels, len(reachable)), np.nan)
    cross_cosines = np.full((n_obs, channels, 4, 4), np.nan)
    effective_rank = np.full((n_obs, channels), np.nan)
    cancellation = np.full((n_obs, channels), np.nan)
    for obs in range(n_obs):
        vectors = np.asarray(array[obs, reachable], dtype=np.float64)
        norms, status, _ = gradient_validity(vectors, absolute_floor, relative_floor)
        gram = np.einsum("sck,tck->cst", vectors, vectors, optimize=True)
        valid = status == "valid"
        all_valid = valid.all(axis=0)
        sums = norms.sum(axis=0)
        fractions[obs, all_valid] = (norms[:, all_valid] / sums[all_valid]).T
        for a, scale_a in enumerate(reachable):
            for b, scale_b in enumerate(reachable):
                pair_valid = valid[a] & valid[b]
                cross_cosines[obs, pair_valid, scale_a, scale_b] = np.clip(
                    gram[pair_valid, a, b] / (norms[a, pair_valid] * norms[b, pair_valid]), -1, 1)
        if all_valid.any():
            unit_gram = gram[all_valid] / (norms[:, all_valid].T[:, :, None] * norms[:, all_valid].T[:, None, :])
            eigenvalues = np.maximum(np.linalg.eigvalsh(unit_gram), 0)
            probabilities = eigenvalues / eigenvalues.sum(axis=1, keepdims=True)
            logp = np.zeros_like(probabilities)
            np.log(probabilities, out=logp, where=probabilities > 0)
            effective_rank[obs, all_valid] = np.exp(-(probabilities * logp).sum(axis=1))
            cancellation[obs, all_valid] = np.sqrt(np.maximum(gram[all_valid].sum(axis=(1, 2)), 0)) / sums[all_valid]
    return fractions, cross_cosines, effective_rank, cancellation


def profile_pca_and_clusters(labeled_case_profiles, unlabeled_case_profiles, coverage_l, coverage_u,
                             seed=142, bootstrap_draws=100, minimum_coverage=.75):
    """Explore norm-share profiles; QC exclusions are not a channel selection rule."""
    left = masked_mean(labeled_case_profiles, axis=0)
    right = masked_mean(unlabeled_case_profiles, axis=0)
    profiles = np.concatenate([left, right], axis=1)
    eligible = ((coverage_l >= minimum_coverage) & (coverage_u >= minimum_coverage)
                & np.isfinite(profiles).all(axis=1))
    indices = np.flatnonzero(eligible)
    common = dict(eligible_channels=indices.tolist(), excluded_channels=np.flatnonzero(~eligible).tolist(),
                  minimum_coverage=minimum_coverage, reachable_scale_count=left.shape[1],
                  algebraic_rank_upper_bound=2 * (left.shape[1]-1),
                  profile_definition="Per-case mean of complete-valid per-patch raw gradient-norm shares; concatenate L and U pools",
                  causal_interpretation="Descriptive grouping only; neither synergy nor complementarity is established")
    if len(indices) < 5:
        return dict(**common, status="INSUFFICIENT_VALID_CHANNELS"), profiles, None, None
    matrix = profiles[indices]
    centered = matrix - matrix.mean(axis=0)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > max(1e-12, (singular[0] if len(singular) else 0) * 1e-10)))
    if rank == 0:
        return dict(**common, status="CONSTANT_PROFILE", rank=0,
                    note="One reachable scale forces a constant unit share; no interpretable PCA/clustering"), profiles, None, None
    explained = singular**2 / (singular**2).sum()
    candidates, labels = [], None
    for k in (2, 3, 4):
        if k >= len(indices):
            continue
        fit = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(matrix)
        unique = len(np.unique(fit.labels_))
        silhouette = float(silhouette_score(matrix, fit.labels_)) if 1 < unique < len(matrix) else None
        candidates.append(dict(k=k, silhouette=silhouette, inertia=float(fit.inertia_),
                               cluster_sizes=np.bincount(fit.labels_, minlength=k).tolist()))
        if k == 2:
            labels = fit.labels_
    generator = np.random.default_rng(seed)
    stability = []
    for _ in range(bootstrap_draws):
        left_indices = generator.integers(0, len(labeled_case_profiles), len(labeled_case_profiles))
        right_indices = generator.integers(0, len(unlabeled_case_profiles), len(unlabeled_case_profiles))
        sample = np.concatenate([masked_mean(labeled_case_profiles[left_indices], axis=0),
                                 masked_mean(unlabeled_case_profiles[right_indices], axis=0)], axis=1)[indices]
        if not np.isfinite(sample).all() or labels is None:
            continue
        fit = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(sample)
        stability.append(adjusted_rand_score(labels, fit.labels_))
    scores = u[:, :2] * singular[:2]
    if scores.shape[1] == 1:
        scores = np.pad(scores, ((0, 0), (0, 1)))
    result = dict(**common, status="OK", rank=rank, singular_values=singular.tolist(),
                  explained_variance_ratio=explained.tolist(), pca_loadings=vt[:2].tolist(),
                  cluster_candidates=candidates, bootstrap_draws_completed=len(stability),
                  k2_case_bootstrap_ARI=describe(stability))
    return result, profiles, scores, labels
