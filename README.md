# AB Testing Lab

An experimentation toolkit where the truth is known, so every method can be checked instead of
trusted.

Most portfolio A/B testing projects run a test on a demo dataset and report a p-value. Nobody can
tell whether the analysis was right. Here the true effect is set by the simulator, so the question
"was my method correct" has an answer, and so does "what breaks it".

## Headline results

| Finding | Number |
|---|---|
| Checking a test 2 times inflates false positives | 5% becomes 7.0% |
| Checking 20 times | 5% becomes **25.8%** |
| Checking 40 times | 5% becomes **29.8%** |
| A calibrated sequential boundary at 40 looks | 6.3%, consistent with 5% |
| Power the standard formula promises at 14,749 per arm | 80% |
| Power actually delivered when users differ | **66%** |
| CUPED variance reduction on a continuous metric | 28.4% |
| CUPED variance reduction on a binary metric | 4.5% |

## The three findings worth reading

### 1. Peeking destroys the false positive rate

If you check a fixed horizon test every day and stop when it crosses p below 0.05, you are not
running a 5 percent test. At 40 looks it fires in **29.8 percent** of experiments where there is
nothing to find.

A boundary calibrated on null simulations fixes it: stricter early (z above 13.3 at the first look),
converging to about 2.10 at the last. Realised rate 6.3 percent, which is within simulation error of
the 5 percent target.

This is the single most common mistake in product analytics, and it is cheap to demonstrate once you
have ground truth.

### 2. The power formula is optimistic because users are not identical

The standard sample size formula assumes every user has the same base rate. Real populations do not,
and heterogeneous probabilities add variance beyond the binomial the formula assumes.

At the 14,749 users per arm the formula recommends for a 1 point lift, measured power is **80.8
percent when users are identical** and **66.2 percent when they vary**. A 14 point shortfall, in the
direction that costs you a failed experiment.

The lesson is not to distrust the formula. It is to validate the plan by simulation whenever the
population is heterogeneous, which is always.

### 3. CUPED belongs on continuous metrics

Pre period adjustment cut variance by **28.4 percent** on a revenue metric and **4.5 percent** on a
conversion metric.

That is not a bug, it is the design. A binary outcome carries little information about a continuous
covariate, so there is almost nothing to adjust away. Using CUPED on a conversion rate and expecting
a third off your sample is a common and expensive misunderstanding.

## Decision framework

A p-value is not a decision. `src/decision.py` requires four things before it recommends shipping:

1. **Sample ratio check.** A broken assignment invalidates everything, whatever the metric says.
2. **Significance and direction** on the primary metric.
3. **A practical threshold**, because a significant 0.2 point lift is not worth shipping.
4. **Guardrails**, each with its own direction, because churn rising is bad while revenue rising is
   good.

Every verdict prints its reasoning, so it can be argued with:

```
DECISION: DO NOT SHIP
  Primary metric improved by +0.0200 (95% CI +0.0150 to +0.0250).
  churn rose by 0.0300 (p=0.004)
  A guardrail metric degraded, so the win is not clean.
```

## Layout

```
src/simulate.py   experiment simulator with a known effect, plus sequential paths
src/stats.py      proportion and Welch tests, power, CUPED, sequential boundaries, multiple testing
src/decision.py   sample ratio check and the ship / no ship framework
scripts/          one acceptance test per layer
scripts/make_report.py   regenerates the report and both charts
reports/          experiment report, peeking chart, power chart
```

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/acceptance_layer1.py   # estimator recovers a known truth
python scripts/acceptance_layer2.py   # false positive rate and power
python scripts/acceptance_layer3.py   # peeking and sequential control
python scripts/acceptance_layer4.py   # CUPED and multiple testing
python scripts/acceptance_layer5.py   # decision framework
python scripts/make_report.py         # report and charts
```

Everything runs in seconds. No dataset, no API keys, no cost.

## What the layers prove

| Layer | Claim | Evidence |
|---|---|---|
| 1 | The estimator is unbiased and its intervals have correct coverage | 94.5% coverage over 400 simulated experiments |
| 2 | The test is calibrated and the power formula is honest | 4.7% false positive rate, formula exact for identical users |
| 3 | Peeking breaks the test and a boundary repairs it | 29.8% naive against 6.3% sequential at 40 looks |
| 4 | CUPED and multiple testing correction work, and have limits | 28.4% variance cut on continuous, 4.5% on binary |
| 5 | Decisions are conservative and explained | 8 of 8 decision cases, including SRM and guardrails |

## What this does not cover

- Novelty and primacy effects, which need a holdout or a longer window.
- Interference between units, in marketplaces, social products, or anything with supply constraints.
- Network effects and switchback designs.
- Bayesian and always valid inference, which are alternatives to the boundary used here.
- Real traffic, which is never as clean as a simulator.

Each limitation is a real one, and each would be the next layer in a production system.
