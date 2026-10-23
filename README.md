# Knowledge-based early warning for digital-venture failure

Short communication and the code that produces its figures, tables, and numbers.

The study asks whether a knowledge base of entrepreneurial and digital-transformation rules, an artificial-intelligence model, or a mixture of the two is the right warning score for failure among digital ventures. The manuscript is `PAPER.md`. Every reported metric comes from `ews.py`.

## Run

```bash
python ews.py
```

That rewrites `fig1_architecture.png`, `fig2_roc.png`, and `results.json`. The random seed is 42. No network connection is required.

## Files

| File | Role |
| --- | --- |
| `ews.py` | Rules, models, cross-validation, figures |
| `venture_data.csv` | 472 public venture profiles, company names removed |
| `PAPER.md` | Short communication with two figures and two tables |
| `fig1_architecture.png` | System diagram |
| `fig2_roc.png` | Out-of-fold ROC curves |

## Data

`venture_data.csv` is a column subset of the public CAX startup table (`CAX_Startup_Data.csv`), with company names dropped. The analysis keeps the 452 firms whose product is digital: machine learning, predictive or prescriptive or speech analytics, big data, or a cloud or platform offer. The label is the public company status, failed versus success. It is not a privately collected field sample, and it has no failure date.

## Manuscript note

Add author names and affiliations before submission. Knowledge-Based Systems limits a short communication to 10 double-spaced pages, including tables and figures. The warning score in this study is the gradient-boosting probability. The rules explain the warning. The mixture of the two probabilities does not improve discrimination on this cohort, and the paper says so.
