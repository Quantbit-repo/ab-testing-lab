"""Decision layer: turning statistics into a ship or no ship recommendation.

A p-value is not a decision. A decision needs four things:

1. A sample ratio check, because a broken assignment makes every other number
   meaningless.
2. A primary metric with a direction and a practical threshold, not just
   significance.
3. Guardrail metrics, because a win on one metric paid for with a loss on
   another is not a win.
4. A stated reason, so the decision can be argued with.

This module is deliberately conservative. It returns "do not ship" more often
than a significance test alone would, which is the correct bias for a product
team.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def srm_check(n_control: int, n_treatment: int, expected_treatment_share: float = 0.5,
              alpha: float = 0.001) -> dict:
    """Sample ratio mismatch: are the arms the size they should be?

    A tiny p-value here means assignment is broken, so the experiment is
    invalid regardless of what the metric says. The threshold is deliberately
    strict, because SRM is a bug rather than a finding.
    """
    n = n_control + n_treatment
    expected = np.array([n * (1 - expected_treatment_share), n * expected_treatment_share])
    observed = np.array([n_control, n_treatment])

    chi2, p_value = sps.chisquare(observed, expected)
    return {
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "control_share": float(n_control / n) if n else 0.0,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "passed": bool(p_value >= alpha),
    }


def decide(primary: dict, guardrails: dict | None = None, srm: dict | None = None,
           min_practical_effect: float = 0.0, alpha: float = 0.05) -> dict:
    """Ship or no ship, with a reason a product manager can act on.

    primary: dict with keys diff, p_value, ci_low, ci_high.
    guardrails: mapping of metric name to dict with the same shape.
    """
    reasons = []

    if srm is not None and not srm.get("passed", True):
        return {
            "decision": "invalid",
            "reasons": [
                f"Sample ratio mismatch: {srm['control_share']:.1%} in control, "
                f"expected {1 - 0.5:.0%}. Assignment is broken, so the result "
                f"cannot be trusted."
            ],
            "ship": False,
        }

    significant = primary["p_value"] < alpha
    positive = primary["diff"] > 0
    practical = primary["diff"] >= min_practical_effect

    if not significant:
        reasons.append(f"No significant effect (p={primary['p_value']:.3f}).")
        return {"decision": "no ship", "reasons": reasons, "ship": False}

    if not positive:
        reasons.append(
            f"Effect is significant but negative ({primary['diff']:+.4f})."
        )
        return {"decision": "no ship", "reasons": reasons, "ship": False}

    if not practical:
        reasons.append(
            f"Effect is significant but below the practical threshold "
            f"({primary['diff']:+.4f} vs {min_practical_effect:+.4f} required)."
        )
        return {"decision": "no ship", "reasons": reasons, "ship": False}

    reasons.append(
        f"Primary metric improved by {primary['diff']:+.4f} "
        f"(95% CI {primary['ci_low']:+.4f} to {primary['ci_high']:+.4f})."
    )

    degraded = []
    for name, result in (guardrails or {}).items():
        # Not every guardrail is better when it rises. Churn, latency and refund
        # rate are all worse upward, so the caller states the direction.
        higher_is_better = result.get("higher_is_better", True)
        worse = result["diff"] < 0 if higher_is_better else result["diff"] > 0
        if result["p_value"] < alpha and worse:
            direction = "fell" if higher_is_better else "rose"
            degraded.append(
                f"{name} {direction} by {abs(result['diff']):.4f} "
                f"(p={result['p_value']:.3f})"
            )

    if degraded:
        reasons.extend(degraded)
        reasons.append("A guardrail metric degraded, so the win is not clean.")
        return {"decision": "do not ship", "reasons": reasons, "ship": False}

    if guardrails:
        reasons.append("All guardrail metrics held.")

    return {"decision": "ship", "reasons": reasons, "ship": True}


def format_decision(result: dict) -> str:
    lines = [f"DECISION: {result['decision'].upper()}"]
    for reason in result["reasons"]:
        lines.append(f"  {reason}")
    return "\n".join(lines)
