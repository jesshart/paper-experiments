from __future__ import annotations

from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ExperimentResult:
    """Container for the simulation outputs."""
    n_voxels: int
    alpha: float
    n_uncorrected_significant: int
    bonferroni_threshold: float
    n_bonferroni_significant: int
    p_values: np.ndarray
    correlations: np.ndarray


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Approximate the standard normal CDF elementwise."""
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def simulate_false_positives(
    n_timepoints: int = 200,
    n_voxels: int = 20_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ExperimentResult:
    """
    Simulate voxel-wise tests under the complete null hypothesis.

    Parameters
    ----------
    n_timepoints
        Number of observations per voxel.
    n_voxels
        Number of independent voxel tests.
    alpha
        Uncorrected significance threshold.
    seed
        Random seed for reproducibility.

    Returns
    -------
    ExperimentResult
        Summary of false positives with and without correction.
    """
    rng = np.random.default_rng(seed)

    # Fake experimental regressor: e.g., "emotional vs non-emotional" blocks
    task = rng.integers(0, 2, size=n_timepoints).astype(float)
    task = (task - task.mean()) / task.std(ddof=1)

    # Pure noise: no real signal in any voxel
    noise = rng.normal(size=(n_timepoints, n_voxels))
    noise = (noise - noise.mean(axis=0)) / noise.std(axis=0, ddof=1)

    # Pearson correlation between each voxel and the fake task
    correlations = (task[:, None] * noise).sum(axis=0) / (n_timepoints - 1)

    # Convert correlation to test statistic
    df = n_timepoints - 2
    t_stats = correlations * np.sqrt(df / np.maximum(1e-12, 1.0 - correlations**2))

    # Large-sample two-sided p-value using normal approximation
    p_values = 2.0 * (1.0 - normal_cdf(np.abs(t_stats)))

    n_uncorrected_significant = int((p_values < alpha).sum())

    bonferroni_threshold = alpha / n_voxels
    n_bonferroni_significant = int((p_values < bonferroni_threshold).sum())

    return ExperimentResult(
        n_voxels=n_voxels,
        alpha=alpha,
        n_uncorrected_significant=n_uncorrected_significant,
        bonferroni_threshold=bonferroni_threshold,
        n_bonferroni_significant=n_bonferroni_significant,
        p_values=p_values,
        correlations=correlations,
    )


def main() -> None:
    result = simulate_false_positives()

    expected_false_positives = result.alpha * result.n_voxels

    print("=== False Positive Simulation ===")
    print(f"Number of voxels tested: {result.n_voxels:,}")
    print(f"Uncorrected alpha: {result.alpha}")
    print(f"Expected false positives by chance alone: ~{expected_false_positives:.0f}")
    print(f"Observed uncorrected significant voxels: {result.n_uncorrected_significant:,}")
    print()
    print(f"Bonferroni threshold: {result.bonferroni_threshold:.2e}")
    print(f"Observed Bonferroni-significant voxels: {result.n_bonferroni_significant:,}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(result.p_values, bins=50, edgecolor="black")
    ax.axvline(result.alpha, linestyle="--", label=f"uncorrected alpha = {result.alpha}")
    ax.axvline(
        result.bonferroni_threshold,
        linestyle="--",
        label=f"Bonferroni = {result.bonferroni_threshold:.2e}",
    )
    ax.set_title("P-value distribution under the null (all voxels are noise)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()