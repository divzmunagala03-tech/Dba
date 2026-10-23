# KG-EWS: knowledge-guided early warning for digital-venture failure

Short communication and the program that produces its cohorts, figures, tables, and tests.

KG-EWS adds eight committee rules to the log-odds of a logistic score. The rules cover capability, innovation, team, runway, and traction in an entrepreneurial firm that is trying to transform digitally. The benchmark is synthetic and seeded. Every number in `PAPER.md` comes from `ews.py`.

## Run

```bash
python3 ews.py
```

The run writes `fig1_architecture.png`, `fig2_results.png`, `cohort_results.csv`, and `results.json`. Primary cohort seeds are 9000 through 9119. A predicate-flip block uses seeds 12000 through 12059. No network connection and no external data file are required.

## Files

| File | Role |
| --- | --- |
| `ews.py` | Generator, rules, models, paired tests, figures |
| `PAPER.md` | Short communication: two figures and two tables |
| `fig1_architecture.png` | Warning card and one rule-triggered test case |
| `fig2_results.png` | Paired cost advantage and the false-alarm tradeoff |
| `cohort_results.csv` | Primary cohorts and the predicate-flip cohorts |

## What the experiment claims

On 120 cohorts, KG-EWS has the lowest decision cost and the highest AUC among logistic regression, gradient boosting, the rules alone, and the combined score. The rules alone alarm on most surviving firms. The gain over logistic regression is small, paired, and absent from the cohorts that set the evidence weight to zero. It remains at a miss weight of 3, is not detectable at a miss weight of 8, and does not survive flipping each predicate with probability 0.15. The data are not a field sample and there is no failure date.

## Manuscript note

Add author names before submission. Knowledge-Based Systems limits a short communication to 10 double-spaced pages, including tables and figures.
