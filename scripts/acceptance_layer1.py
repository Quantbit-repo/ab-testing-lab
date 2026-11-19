"""Layer 1 acceptance: the estimator must recover a known truth.

Acceptance criteria:
1. With a known lift, the estimate is close to the truth on average (unbiased).
2. The 95 percent interval contains the truth in roughly 95 percent of runs.
   This is the property that makes an interval meaningful, and it is the one
   nobody checks in a notebook.
3. The same holds under a true null, where coverage should also be near 95.
4. The interval narrows as sample size grows, at roughly the square root rate.

Run: python scripts/acceptance_layer1.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.simulate import simulate  # noqa: E402
from src.stats import two_proportion_test  # noqa: E402

SIMS = 400


def run_case(true_lift: float, n_per_arm: int, seed0: int):
    hits, estimates, widths = 0, [], []
    for i in range(SIMS):
        data = simulate(n_per_arm, 0.10, true_lift, seed=seed0 + i)
        res = two_proportion_test(data["outcome"], data["treatment"])
        estimates.append(res["diff"])
        widths.append(res["ci_high"] - res["ci_low"])
        if res["ci_low"] <= true_lift <= res["ci_high"]:
            hits += 1
    return hits / SIMS, float(np.mean(estimates)), float(np.mean(widths))


def main() -> int:
    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    coverage, mean_est, width = run_case(0.01, 20000, 1000)
    record("interval covers the true lift in about 95 percent of runs",
           0.92 <= coverage <= 0.98, f"coverage {coverage:.1%}")
    record("estimate is close to the truth on average",
           abs(mean_est - 0.01) < 0.0015, f"mean estimate {mean_est:.5f} vs true 0.01000")

    null_coverage, null_mean, _ = run_case(0.0, 20000, 5000)
    record("coverage holds under a true null",
           0.92 <= null_coverage <= 0.98, f"coverage {null_coverage:.1%}")
    record("null estimate is centred on zero",
           abs(null_mean) < 0.0015, f"mean estimate {null_mean:+.5f}")

    _, _, wide = run_case(0.01, 5000, 9000)
    _, _, narrow = run_case(0.01, 20000, 9000)
    ratio = wide / narrow
    record("interval narrows with sample size at roughly the square root rate",
           1.7 <= ratio <= 2.3, f"width ratio {ratio:.2f} for a 4x sample increase")

    print(f"{'CHECK':62} {'RESULT':7} DETAIL")
    print("-" * 108)
    passed = 0
    for name, ok, detail in checks:
        print(f"{name:62} {'PASS' if ok else 'FAIL':7} {detail}")
        passed += int(ok)
    print("-" * 108)
    print(f"=== LAYER 1 ACCEPTANCE: {passed}/{len(checks)} passed ===")
    print(f"\n{SIMS} simulated experiments per case, 20,000 users per arm, 10% base rate")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
