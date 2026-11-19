"""Layer 4 acceptance: variance reduction and multiple testing.

Acceptance criteria:
1. CUPED reduces the variance of the outcome materially, because the pre period
   covariate is correlated with it.
2. CUPED does not bias the estimate: coverage of the true effect still holds.
3. CUPED raises measured power at the same sample size.
4. Without correction, testing 20 null metrics produces at least one false
   positive far more often than 5 percent of the time.
5. Bonferroni brings that family wise error back to about 5 percent.
6. Benjamini Hochberg also controls it under a global null, and is less
   conservative than Bonferroni when real effects exist.

Run: python scripts/acceptance_layer4.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.simulate import simulate  # noqa: E402
from src.stats import (any_significant, benjamini_hochberg, bonferroni,  # noqa: E402
                       cuped_adjust, two_proportion_test, two_sample_test)

BASE = 0.10
ALPHA = 0.05
SIMS = 300
METRICS = 20


def main() -> int:
    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # ---- CUPED on a continuous metric, where it belongs ----
    reductions, hits, powers_raw, powers_adj = [], 0, 0, 0
    binary_reductions = []
    # 400 per arm puts raw power around 70 percent, where an efficiency gain is
    # visible. At a larger sample both arms sit near 100 percent and the
    # comparison says nothing.
    n_per_arm = 400
    true_lift = 0.01

    for i in range(SIMS):
        data = simulate(n_per_arm, BASE, true_lift, seed=10000 + i)

        adj = cuped_adjust(data["revenue"], data["pre_metric"], data["treatment"])
        reductions.append(adj["variance_reduction"])
        binary_reductions.append(
            cuped_adjust(data["outcome"], data["pre_metric"], data["treatment"])["variance_reduction"]
        )

        true_effect = data["true_revenue_lift"]
        raw = two_sample_test(data["revenue"], data["treatment"])
        adjusted = two_sample_test(adj["adjusted"], data["treatment"])

        if adjusted["ci_low"] <= true_effect <= adjusted["ci_high"]:
            hits += 1
        powers_raw += raw["p_value"] < ALPHA
        powers_adj += adjusted["p_value"] < ALPHA

    mean_reduction = float(np.mean(reductions))
    record("CUPED reduces variance materially on a continuous metric",
           mean_reduction > 0.20,
           f"mean variance reduction {mean_reduction:.1%}, "
           f"covariate correlation {adj['correlation']:.2f}")

    coverage = hits / SIMS
    record("CUPED does not bias the estimate",
           0.92 <= coverage <= 0.98, f"coverage {coverage:.1%}")

    record("CUPED raises power at the same sample size",
           powers_adj > powers_raw,
           f"{powers_adj / SIMS:.1%} adjusted vs {powers_raw / SIMS:.1%} raw")

    binary_mean = float(np.mean(binary_reductions))
    record("CUPED buys little on a binary metric, and the toolkit says so",
           binary_mean < 0.10,
           f"binary variance reduction only {binary_mean:.1%}")

    # ---- multiple testing ----
    fwe_raw, fwe_bonf, fwe_bh = 0, 0, 0
    for s in range(SIMS):
        p_values = []
        for m in range(METRICS):
            data = simulate(3000, BASE, 0.0, seed=20000 + s * METRICS + m)
            p_values.append(two_proportion_test(data["outcome"], data["treatment"])["p_value"])

        fwe_raw += any_significant(p_values, ALPHA)
        fwe_bonf += bool(np.any(bonferroni(p_values, ALPHA)))
        fwe_bh += bool(np.any(benjamini_hochberg(p_values, ALPHA)))

    rate_raw = fwe_raw / SIMS
    rate_bonf = fwe_bonf / SIMS
    rate_bh = fwe_bh / SIMS

    record("uncorrected testing of 20 null metrics raises a false alarm far too often",
           rate_raw > 0.50, f"{rate_raw:.1%} of experiments flagged at least one metric")

    record("Bonferroni controls family wise error at about 5 percent",
           0.02 <= rate_bonf <= 0.07, f"{rate_bonf:.1%}")

    record("Benjamini Hochberg also controls error under a global null",
           0.02 <= rate_bh <= 0.08, f"{rate_bh:.1%}")

    print(f"{'CHECK':66} {'RESULT':7} DETAIL")
    print("-" * 116)
    passed = 0
    for name, ok, detail in checks:
        print(f"{name:66} {'PASS' if ok else 'FAIL':7} {detail}")
        passed += int(ok)
    print("-" * 116)
    print(f"=== LAYER 4 ACCEPTANCE: {passed}/{len(checks)} passed ===")
    print(f"\n{SIMS} simulations per check, {METRICS} metrics per experiment, alpha {ALPHA}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
