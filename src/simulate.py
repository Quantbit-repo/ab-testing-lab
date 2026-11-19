"""Experiment simulator with a known ground truth.

The whole point of this project: in a real experiment you never learn whether
your analysis was right, only whether it was convincing. Here you set the true
effect yourself, so every method can be checked against the answer.

Two kinds of simulation:

1. `simulate` returns one experiment, with per user covariates and outcomes.
2. `simulate_paths` returns the running statistics at K looks, which is what
   you need to study peeking.

The covariate exists so CUPED has something to work with. It is drawn first,
correlated with the latent propensity, and independent of assignment. Using a
post treatment variable would leak the treatment effect into the adjustment.
"""

from __future__ import annotations

import numpy as np

# Latent propensity drives both the pre-period metric and the outcome, which is
# what makes the covariate informative. 0.6 gives a moderate, realistic
# correlation rather than an unrealistic near 1.
DEFAULT_RHO = 0.6
DEFAULT_BETA = 1.2


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1.0 - p))


def simulate(n_per_arm: int, base_rate: float = 0.10, absolute_lift: float = 0.0,
             rho: float = DEFAULT_RHO, beta: float = DEFAULT_BETA,
             seed: int = 0, assignment_ratio: float = 0.5) -> dict:
    """Simulate one two arm experiment.

    absolute_lift is the change in probability for treatment, in absolute terms
    (0.01 is one percentage point). 0.0 is a true null.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_arm

    latent = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)
    pre_metric = rho * latent + np.sqrt(1 - rho ** 2) * noise

    # Calibrated so the control arm averages approximately base_rate.
    p_control = _sigmoid(_logit(base_rate) + beta * latent)
    in_treatment = rng.random(n) < assignment_ratio
    p = np.clip(p_control + np.where(in_treatment, absolute_lift, 0.0), 1e-6, 1 - 1e-6)

    outcome = (rng.random(n) < p).astype(float)

    # A continuous metric, which is where CUPED earns its keep. Binary outcomes
    # correlate weakly with a covariate, so adjustment buys almost nothing there.
    revenue = (10.0 + 5.0 * latent + 2.5 * rng.normal(0.0, 1.0, n)
               + np.where(in_treatment, absolute_lift * 100.0, 0.0))

    return {
        "treatment": in_treatment.astype(int),
        "pre_metric": pre_metric,
        "outcome": outcome,
        "revenue": revenue,
        "true_lift": absolute_lift,
        "true_revenue_lift": float(absolute_lift * 100.0),
        "base_rate": base_rate,
    }


def simulate_paths(n_per_arm: int, base_rate: float = 0.10,
                   absolute_lift: float = 0.0, looks: int = 20,
                   rho: float = DEFAULT_RHO, beta: float = DEFAULT_BETA,
                   seed: int = 0) -> dict:
    """Simulate the running statistics at `looks` evenly spaced points.

    Users are assumed to arrive in random order, so each look is a random
    subsample of the final population. That is the friendly case: if anything,
    real traffic is more correlated than this and peeking is worse.
    """
    data = simulate(n_per_arm, base_rate, absolute_lift, rho, beta, seed)
    n = len(data["outcome"])
    order = np.random.default_rng(seed + 1).permutation(n)

    sizes = (np.arange(1, looks + 1) / looks * n).astype(int)

    t = data["treatment"][order].astype(float)
    y = data["outcome"][order].astype(float)

    # Cumulative sums make each look O(1) instead of re slicing the arrays.
    # With thousands of simulated experiments this is the difference between
    # a minute and an hour.
    cum_n1 = np.cumsum(t)
    cum_s1 = np.cumsum(y * t)
    cum_n0 = np.cumsum(1.0 - t)
    cum_s0 = np.cumsum(y * (1.0 - t))

    idx = sizes - 1
    n1, n0 = cum_n1[idx], cum_n0[idx]
    p1 = cum_s1[idx] / np.maximum(n1, 1)
    p0 = cum_s0[idx] / np.maximum(n0, 1)

    diffs = p1 - p0
    ses = np.sqrt(p1 * (1 - p1) / np.maximum(n1, 1) + p0 * (1 - p0) / np.maximum(n0, 1))
    ses = np.maximum(ses, 1e-12)

    return {
        "looks": looks,
        "diffs": diffs,
        "ses": ses,
        "sizes": sizes,
        "true_lift": absolute_lift,
        "z": diffs / ses,
    }
