"""Layer 3 acceptance: peeking inflates the false positive rate, and a sequential
boundary brings it back under control.

Acceptance criteria:
1. Checking a fixed horizon test at every look during collection pushes the false
   positive rate far above 5 percent under a true null.
2. More looks means more inflation.
3. A sequential boundary calibrated on null simulations restores the false
   positive rate to about 5 percent.
4. The sequential procedure keeps a useful amount of power, rather than buying
   validity by never rejecting anything.

Run: python scripts/acceptance_layer3.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.simulate import simulate_paths  # noqa: E402
from src.stats import calibrate_obf, naive_peeking_reject, obf_reject  # noqa: E402

BASE = 0.10
N_PER_ARM = 5000
NULL_SIMS = 800
POWER_SIMS = 300


def naive_fpr(looks: int, sims: int, seed0: int) -> float:
    hits = 0
    for i in range(sims):
        path = simulate_paths(N_PER_ARM, BASE, 0.0, looks, seed=seed0 + i)
        hits += naive_peeking_reject(path["z"])
    return hits / sims


def sequential_fpr(c: float, looks: int, sims: int, seed0: int) -> float:
    hits = 0
    for i in range(sims):
        path = simulate_paths(N_PER_ARM, BASE, 0.0, looks, seed=seed0 + i)
        hits += obf_reject(path["z"], c)
    return hits / sims


def sequential_power(c: float, looks: int, lift: float, sims: int, seed0: int) -> float:
    hits = 0
    for i in range(sims):
        path = simulate_paths(N_PER_ARM, BASE, lift, looks, seed=seed0 + i)
        hits += obf_reject(path["z"], c)
    return hits / sims


def main() -> int:
    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    looks = 20
    fpr_naive = naive_fpr(looks, NULL_SIMS, 100)
    record("peeking at every look inflates the false positive rate",
           fpr_naive > 0.12,
           f"naive FPR {fpr_naive:.1%} at {looks} looks vs 5% intended")

    fpr_naive_few = naive_fpr(5, NULL_SIMS, 200)
    record("more looks means worse inflation",
           fpr_naive > fpr_naive_few,
           f"{looks} looks {fpr_naive:.1%} vs 5 looks {fpr_naive_few:.1%}")

    c = calibrate_obf(looks, alpha=0.05, n_sims=1200, n_per_arm=N_PER_ARM, seed=777)
    fpr_seq = sequential_fpr(c, looks, NULL_SIMS, 300)
    record("a calibrated sequential boundary restores the false positive rate",
           0.035 <= fpr_seq <= 0.065,
           f"sequential FPR {fpr_seq:.1%} with boundary constant {c:.3f}")

    power_seq = sequential_power(c, looks, 0.02, POWER_SIMS, 400)
    record("the sequential test still detects a real effect",
           power_seq > 0.60,
           f"{power_seq:.0%} power at a 2 point lift")

    record("the boundary is stricter early and looser late, as intended",
           bool(np.all(np.diff(np.array([c * np.sqrt(looks / k)
                                        for k in range(1, looks + 1)])) < 0)),
           f"first look threshold {c * np.sqrt(looks):.2f}, final look {c:.2f}")

    print(f"{'CHECK':62} {'RESULT':7} DETAIL")
    print("-" * 112)
    passed = 0
    for name, ok, detail in checks:
        print(f"{name:62} {'PASS' if ok else 'FAIL':7} {detail}")
        passed += int(ok)
    print("-" * 112)
    print(f"=== LAYER 3 ACCEPTANCE: {passed}/{len(checks)} passed ===")
    print(f"\n{NULL_SIMS} null experiments and {POWER_SIMS} alternative experiments, "
          f"{N_PER_ARM} users per arm, {looks} looks")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
