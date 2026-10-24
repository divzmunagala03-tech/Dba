# KG-EWS: knowledge-guided early warning for digital-venture failure

Short communication and the program that produces its cohorts, figures, tables, and tests.

The manuscript is in this repository:

- [PAPER.md](PAPER.md) — the manuscript, readable on GitHub
- [manuscript.pdf](manuscript.pdf) — the same manuscript as a PDF
- [manuscript.docx](manuscript.docx) — the same manuscript as a Word file

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
| `PAPER.md` | Manuscript source: two figures and two tables |
| `manuscript.pdf` | The same manuscript as a PDF |
| `manuscript.docx` | The same manuscript as a Word file |
| `highlights.txt` | Separate highlights file for journal upload |
| `fig1_architecture.png` | Rule-evidence graph for one test failure |
| `fig2_results.png` | Paired cost advantage and the false-alarm tradeoff |
| `cohort_results.csv` | Primary cohorts and the predicate-flip cohorts |

## What the experiment claims

On 120 cohorts, KG-EWS has the lowest decision cost and the highest AUC among logistic regression, gradient boosting, the rules alone, and the combined score. The rules alone alarm on most surviving firms. The gain over logistic regression is small, paired, and absent from the cohorts that set the evidence weight to zero. It remains at a miss weight of 3, is not detectable at a miss weight of 8, and does not survive flipping each predicate with probability 0.15. The data are not a field sample and there is no failure date.

## Highlights

Knowledge-Based Systems asks for highlights as a separate editable file whose name contains “highlights.” That file is [highlights.txt](highlights.txt). Each bullet is at most 85 characters.

- KG-EWS adds committee-rule evidence to a logistic warning score.
- Validation sets the evidence weight from a five-to-one miss cost.
- Across 120 cohorts the paired cost gain over logistic regression is 0.024.
- That gain remains at a three-to-one cost and is lost if predicates are flipped.
- Rules alone catch failures by alarming on most surviving firms.

## Manuscript note

The formatted manuscript follows the short-communication layout of the reference document: title, article type, author, abstract, keywords, numbered sections, table notes, figure captions, declarations, and author-year references. Affiliation, e-mail, and full postal address are still required before submission. Knowledge-Based Systems limits a short communication to 10 double-spaced pages, including tables and figures.
