"""Layer 5 acceptance: the decision layer must reach the right verdict.

Acceptance criteria:
1. A clear win ships.
2. A clear loss does not ship.
3. No effect does not ship.
4. A significant but practically trivial effect does not ship.
5. A win that damages a guardrail metric does not ship.
6. A broken sample ratio makes the result invalid, whatever the metric says.
7. Every decision carries a stated reason.

Run: python scripts/acceptance_layer5.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.decision import decide, format_decision, srm_check  # noqa: E402


def result(diff, p_value, half_width=0.005):
    return {"diff": diff, "p_value": p_value,
            "ci_low": diff - half_width, "ci_high": diff + half_width}


def main() -> int:
    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    good_srm = srm_check(50000, 49800)

    win = decide(result(0.020, 0.0005), srm=good_srm)
    record("a clear win ships", win["decision"] == "ship", win["decision"])

    loss = decide(result(-0.020, 0.0005), srm=good_srm)
    record("a clear loss does not ship", loss["decision"] == "no ship", loss["decision"])

    flat = decide(result(0.0005, 0.62), srm=good_srm)
    record("no effect does not ship", flat["decision"] == "no ship", flat["decision"])

    trivial = decide(result(0.002, 0.010), srm=good_srm, min_practical_effect=0.010)
    record("a statistically significant but tiny effect does not ship",
           trivial["decision"] == "no ship",
           f"{trivial['decision']}, {trivial['reasons'][0][:60]}")

    # churn rising is bad, so this guardrail is worse when it goes up
    harmed = decide(result(0.020, 0.0005),
                    guardrails={"churn": {**result(0.030, 0.004),
                                          "higher_is_better": False}},
                    srm=good_srm)
    record("a win that damages a guardrail does not ship",
           harmed["decision"] == "do not ship",
           f"{harmed['decision']}, {harmed['reasons'][-1][:50]}")

    broken = srm_check(50000, 45000)
    invalid = decide(result(0.020, 0.0005), srm=broken)
    record("a broken sample ratio invalidates the result",
           invalid["decision"] == "invalid",
           f"control share {broken['control_share']:.1%}, p={broken['p_value']:.1e}")

    record("a healthy sample ratio passes the check",
           good_srm["passed"], f"control share {good_srm['control_share']:.1%}")

    all_have_reasons = all(r["reasons"] for r in (win, loss, flat, trivial, harmed, invalid))
    record("every decision states a reason", all_have_reasons,
           f"{len(win['reasons'])} reasons on the win case")

    print(f"{'CHECK':68} {'RESULT':7}")
    print("-" * 88)
    passed = 0
    for name, ok, detail in checks:
        print(f"{name:68} {'PASS' if ok else 'FAIL':7} {detail}")
        passed += int(ok)
    print("-" * 88)
    print(f"=== LAYER 5 ACCEPTANCE: {passed}/{len(checks)} passed ===")
    print("\nExample output for the guardrail case:")
    print(format_decision(harmed))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
