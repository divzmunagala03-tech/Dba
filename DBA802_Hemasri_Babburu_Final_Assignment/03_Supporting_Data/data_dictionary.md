# Data dictionary

`cohort_results.csv` has one row per cohort. `block` is `main` (120 rows, seeds 9000–9119) or `stress` (60 rows, seeds 12000–12059).

| Column | Meaning |
| --- | --- |
| cohort | Cohort index inside its block |
| lambda | Evidence weight chosen on the validation slice at miss weight 5. Grid {0, 0.2, …, 3.0} |
| failure_rate | Share of ventures with y = 1 |
| rule_failure_share | Share of failures for which at least one of R1–R7 fires |
| lin_auc, kb_auc, ai_auc, kg_auc | Test AUC for logistic regression, knowledge base, gradient boosting, and KG-EWS |
| lin_j, kb_j, ai_j, kg_j | Test decision cost at miss weight 5 |
| *_miss_rule | Share of rule-triggered failures with no alert |
| *_miss_smooth | Share of other failures with no alert |
| *_false_alarm | False-alarm rate among surviving firms |
| *_justified | Share of alerts that land on a rule-triggered firm. Not a column in the manuscript tables |
| lambda_w3, lin_j_w3, kg_j_w3 | Evidence weight and costs when the miss weight is re-set to 3. Models are not refit |
| lambda_w8, lin_j_w8, kg_j_w8 | Same at miss weight 8 |
| block | `main` or `stress` |

Prefixes: `lin` logistic regression, `kb` knowledge base, `ai` gradient boosting, `kg` KG-EWS.

Venture-level rows are not stored. The generator redraws them from the seeds. The ten operating fields are runway, burn, digital investment, capability, innovation throughput, team digital experience, market turbulence, customer concentration, pivot count, and adoption lag.

Noise scales, in that field order: 0.80, 0.15, 0.18, 0.18, 0.18, 0.18, 0.04, 0.04, 0.70, 0.18.

Expert evidence weights, R1 through P1: 1.00, 0.85, 0.80, 0.95, 0.60, 0.55, 0.75, −1.05. These are not the coefficients in Equation (4).
