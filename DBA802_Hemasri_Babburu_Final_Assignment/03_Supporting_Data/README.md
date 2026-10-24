# Supporting data for KG-EWS

This folder holds the data and code that produce the manuscript. No field dataset was used, and none has been invented here.

## 1. What data were used

Each cohort is a synthetic draw of 380 ventures. The primary benchmark is 120 cohorts. A predicate-flip check uses 60 further cohorts. The stored table is `cohort_results.csv` (180 rows). The stored summary is `results.json`.

## 2. Where the data originated

`ews.py` generates every venture. Cohort k in the primary block uses seed 9000 + k. The stress block uses seeds 12000 through 12059. There is no external CSV, no Crunchbase extract, and no archived dataset DOI.

The failure draw uses the log-odds in Equation (4) of the manuscript. The eight committee rules are coded on the clean operating state. Logistic regression and gradient boosting see a noisy copy of that state.

## 3. How the data were prepared

There is no cleaning step and no row was dropped. The script draws the operating state, applies the rules, draws the failure label, adds the documented measurement noise, splits each cohort, fits the models, and chooses the evidence weight and the alert threshold on the validation slice.

The split is stratified: 30% test, then 25% of the remainder as validation. That is a fit slice of 52.5% and a validation slice of 17.5%.

## 4. Which scripts analyze the data

`ews.py` is the only analysis script. It fits logistic regression, gradient boosting, the knowledge-base logistic, and KG-EWS. It writes the cohort table, the summary JSON, and the results figure.

## 5. How the results can be reproduced

From the research repository root, with the pinned libraries in `requirements.txt`:

```bash
python3 ews.py
```

The run is seeded. It rewrites `cohort_results.csv`, `results.json`, and `fig1_results.png` beside the script. This folder is a copy for the assignment. Running this copy would write those outputs into this folder. The numbers checked against the manuscript are the copies already stored here.

## 6. Restricted or unavailable files

No file in this study is restricted. What does not exist, and is not included:

- A field sample of entrepreneurial firms.
- A failure date or a calendar lead time.
- An external data dictionary from a vendor.
- A journal submission confirmation screenshot.

`results.json` is gitignored in the research repository so the repository stays small. It is included here because it is the summary the manuscript was checked against.
