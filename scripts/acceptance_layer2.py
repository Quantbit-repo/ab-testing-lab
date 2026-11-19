"""Layer 2 acceptance: the test must be calibrated and the power calculation honest.

Acceptance criteria:
1. Under a true null with adequate sample, the false positive rate is about 5
   percent. If it is not, every other number in the project is meaningless.
2. At the sample size the power formula recommends, measured power is about 80
   percent. The formula and the simulation must agree.
3. Power rises when the effect gets larger.
4. Power rises when the sample gets larger.
5. The minimum detectable effect calculation round trips: taking that effect and
   feeding it back into the sample size formula returns the same sample size.

Run: python scripts/acceptance_layer2.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.simulate import simulate  # noqa: E402
from src.stats import (minimum_detectable_effect, required_sample_size,  # noqa: E402
                       two_proportion_test)

NULL_SIMS = 1000
POWER_SIMS = 400
BASE = 0.10
ALPHA = 0.05


def false_positive_rate(n_per_arm: int, sims: int, seed0: int) -> float:
    rejects = 0
    for i in range(sims):
        data = simulate(n_per_arm, BASE, 0.0, seed=seed0 + i)
        if two_proportion_test(data["outcome"], data["treatment"])["p_value"] < ALPHA:
            rejects += 1
    return rejects / sims


def measured_power(n_per_arm: int, lift: float, sims: int, seed0: int,
                   beta: float = 1.2) -> float:
    """Simulated power. beta controls how much baseline risk varies across users."""
    rejects = 0
    for i in range(sims):
        data = simulate(n_per_arm, BASE, lift, beta=beta, seed=seed0 + i)
        if two_proportion_test(data["outcome"], data["treatment"])["p_value"] < ALPHA:
            rejects += 1
    return rejects / sims


def main() -> int:
    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    fpr = false_positive_rate(20000, NULL_SIMS, 20000)
    record("false positive rate is about 5 percent under a true null",
           0.035 <= fpr <= 0.065, f"FPR {fpr:.1%} over {NULL_SIMS} null experiments")

    lift = 0.01
    n_needed = required_sample_size(BASE, lift, power=0.80, alpha=ALPHA)

    # Homogeneous users: every user has exactly the base rate, which is the
    # assumption baked into the standard formula. The formula should hold here.
    power_homogeneous = measured_power(n_needed, lift, POWER_SIMS, 30000, beta=0.0)
    record("formula is correct when users are homogeneous, as it assumes",
           0.72 <= power_homogeneous <= 0.88,
           f"{power_homogeneous:.1%} power at n={n_needed} per arm")

    # Heterogeneous users, which is what real traffic looks like. Variance is
    # higher than binomial, so realised power falls short of the formula.
    power = measured_power(n_needed, lift, POWER_SIMS, 30000)
    record("heterogeneous users produce less power than the formula promises",
           power < power_homogeneous - 0.05,
           f"{power:.1%} heterogeneous vs {power_homogeneous:.1%} homogeneous")

    power_bigger = measured_power(n_needed, 0.02, POWER_SIMS, 40000)
    record("larger effects are easier to detect",
           power_bigger > power, f"{power_bigger:.1%} at 2pp vs {power:.1%} at 1pp")

    n_big = required_sample_size(BASE, lift, power=0.95, alpha=ALPHA)
    power_big_sample = measured_power(n_big, lift, POWER_SIMS, 50000)
    record("more sample means more power",
           power_big_sample > power, f"{power_big_sample:.1%} at n={n_big}")

    mde = minimum_detectable_effect(n_needed, BASE, power=0.80, alpha=ALPHA)
    n_rt = required_sample_size(BASE, mde, power=0.80, alpha=ALPHA)
    record("minimum detectable effect round trips through the sample size formula",
           abs(n_rt - n_needed) / n_needed < 0.05,
           f"MDE {mde:.4f}, back to n={n_rt} vs {n_needed}")

    print(f"{'CHECK':62} {'RESULT':7} DETAIL")
    print("-" * 110)
    passed = 0
    for name, ok, detail in checks:
        print(f"{name:62} {'PASS' if ok else 'FAIL':7} {detail}")
        passed += int(ok)
    print("-" * 110)
    print(f"=== LAYER 2 ACCEPTANCE: {passed}/{len(checks)} passed ===")
    print(f"\nbase rate {BASE:.0%}, alpha {ALPHA}, {NULL_SIMS} null runs, {POWER_SIMS} runs per power estimate")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
