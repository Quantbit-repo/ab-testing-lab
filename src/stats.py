"""Statistical machinery, written so each piece can be checked against a known truth.

Everything here is deliberately explicit rather than imported from a black box.
An experiment toolkit that cannot show its false positive rate is not a toolkit,
it is a hope.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------- basic test --

def two_proportion_test(outcome, treatment) -> dict:
    """Difference in means with a normal approximation interval."""
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=int)
    treated, control = y[t == 1], y[t == 0]
    n1, n0 = len(treated), len(control)

    p1, p0 = treated.mean(), control.mean()
    diff = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    if se == 0:
        return {"diff": 0.0, "se": 0.0, "z": 0.0, "p_value": 1.0,
                "ci_low": 0.0, "ci_high": 0.0, "n_control": n0, "n_treatment": n1}

    z = diff / se
    p_value = 2 * (1 - sps.norm.cdf(abs(z)))
    return {
        "diff": float(diff),
        "se": float(se),
        "z": float(z),
        "p_value": float(p_value),
        "ci_low": float(diff - 1.96 * se),
        "ci_high": float(diff + 1.96 * se),
        "n_control": int(n0),
        "n_treatment": int(n1),
    }


def two_sample_test(outcome, treatment) -> dict:
    """Welch test for a continuous metric.

    The proportion test above assumes binary outcomes and uses a Bernoulli
    standard error. Handing it a revenue column would silently understate the
    variance, so continuous metrics go through this instead.
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=int)
    treated, control = y[t == 1], y[t == 0]
    n1, n0 = len(treated), len(control)

    m1, m0 = treated.mean(), control.mean()
    v1, v0 = treated.var(ddof=1), control.var(ddof=1)
    diff = m1 - m0
    se = np.sqrt(v1 / n1 + v0 / n0)
    if se == 0:
        return {"diff": 0.0, "se": 0.0, "z": 0.0, "p_value": 1.0,
                "ci_low": 0.0, "ci_high": 0.0, "n_control": n0, "n_treatment": n1}

    # Welch Satterthwaite degrees of freedom
    df = (v1 / n1 + v0 / n0) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1))
    t_stat = diff / se
    p_value = 2 * (1 - sps.t.cdf(abs(t_stat), df))
    crit = sps.t.ppf(0.975, df)

    return {
        "diff": float(diff),
        "se": float(se),
        "z": float(t_stat),
        "p_value": float(p_value),
        "ci_low": float(diff - crit * se),
        "ci_high": float(diff + crit * se),
        "n_control": int(n0),
        "n_treatment": int(n1),
    }


# ------------------------------------------------------------------- power ---

def required_sample_size(p0: float, absolute_lift: float, power: float = 0.80,
                         alpha: float = 0.05) -> int:
    """Users per arm for a two sided test of a difference in proportions."""
    p1 = min(max(p0 + absolute_lift, 1e-6), 1 - 1e-6)
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_beta = sps.norm.ppf(power)
    delta = abs(absolute_lift)
    if delta == 0:
        raise ValueError("lift must be non zero")
    numerator = (z_alpha + z_beta) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1))
    return int(np.ceil(numerator / delta ** 2))


def minimum_detectable_effect(n_per_arm: int, p0: float, power: float = 0.80,
                              alpha: float = 0.05) -> float:
    """Smallest absolute lift this sample size can detect."""
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_beta = sps.norm.ppf(power)

    def power_at(delta):
        p1 = p0 + delta
        se = np.sqrt(p0 * (1 - p0) / n_per_arm + p1 * (1 - p1) / n_per_arm)
        return 1 - sps.norm.cdf(z_alpha - delta / se)

    lo, hi = 1e-5, 0.5
    for _ in range(80):
        mid = (lo + hi) / 2
        if power_at(mid) < power:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


# ------------------------------------------------------------------- CUPED ---

def cuped_adjust(outcome, pre_metric, treatment) -> dict:
    """Covariate adjustment using pre experiment data.

    theta uses the pooled sample. Assignment is random and the covariate is
    pre treatment, so pooling does not leak the treatment effect.
    """
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(pre_metric, dtype=float)
    t = np.asarray(treatment, dtype=int)

    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    adjusted = y - theta * (x - x.mean())

    var_before = float(np.var(y, ddof=1))
    var_after = float(np.var(adjusted, ddof=1))
    reduction = 1 - var_after / var_before

    return {
        "theta": theta,
        "adjusted": adjusted,
        "variance_before": var_before,
        "variance_after": var_after,
        "variance_reduction": float(reduction),
        "correlation": float(np.corrcoef(y, x)[0, 1]),
    }


# -------------------------------------------------------------- sequential ---

def naive_peeking_reject(z_path, alpha: float = 0.05) -> bool:
    """Reject if any look crosses the fixed horizon threshold. This is the bug."""
    threshold = sps.norm.ppf(1 - alpha / 2)
    return bool(np.any(np.abs(z_path) > threshold))


def obf_thresholds(looks: int, c: float) -> np.ndarray:
    """O'Brien Fleming style boundary: stricter early, converging to c at the end."""
    k = np.arange(1, looks + 1)
    return c * np.sqrt(looks / k)


def obf_reject(z_path, c: float) -> bool:
    thresholds = obf_thresholds(len(z_path), c)
    return bool(np.any(np.abs(z_path) > thresholds))


def calibrate_obf(looks: int, alpha: float = 0.05, n_sims: int = 2000,
                  n_per_arm: int = 2000, base_rate: float = 0.10,
                  seed: int = 123) -> float:
    """Choose the boundary constant so the overall false positive rate is alpha."""
    maxima = []
    for i in range(n_sims):
        path = _null_path(n_per_arm, base_rate, looks, seed + i)
        maxima.append(np.max(np.abs(path) / np.sqrt(looks / np.arange(1, looks + 1))))
    return float(np.quantile(maxima, 1 - alpha))


def _null_path(n_per_arm, base_rate, looks, seed):
    from src.simulate import simulate_paths

    return simulate_paths(n_per_arm, base_rate, 0.0, looks, seed=seed)["z"]


# --------------------------------------------------------- multiple testing --

def bonferroni(p_values, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    return p < (alpha / len(p))


def benjamini_hochberg(p_values, alpha: float = 0.05) -> np.ndarray:
    """False discovery rate control. Returns a boolean rejection mask."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)

    k = np.max(np.where(below)[0])
    cutoff = ranked[k]
    return p <= cutoff


def any_significant(p_values, alpha: float = 0.05) -> bool:
    return bool(np.any(np.asarray(p_values) < alpha))
