# KG-EWS: knowledge-guided early warning for digital-venture failure

Short communication and the program that produces its cohorts, figures, tables, and tests.

The manuscript is in this repository:

- [PAPER.md](PAPER.md) — the manuscript, readable on GitHub
- [manuscript.docx](manuscript.docx) — the Word file to upload as the manuscript
- [manuscript.pdf](manuscript.pdf) — a PDF preview of the same manuscript
- [Highlights.docx](Highlights.docx) — the highlights file to upload
- [Declaration-of-competing-interests.docx](Declaration-of-competing-interests.docx) — the competing-interests file the form requires

KG-EWS adds eight committee rules to the log-odds of a logistic score. The rules cover capability, innovation, team, runway, and traction in an entrepreneurial firm that is trying to transform digitally. The benchmark is synthetic and seeded. Every number in `PAPER.md` comes from `ews.py`.

## Run

```bash
python3 ews.py
```

The run writes `fig1_results.png`, `cohort_results.csv`, and `results.json`. Primary cohort seeds are 9000 through 9119. A predicate-flip block uses seeds 12000 through 12059. No network connection and no external data file are required.

## Files

| File | Role |
| --- | --- |
| `ews.py` | Generator, rules, models, paired tests, figures |
| `PAPER.md` | Manuscript source: one figure and two tables |
| `manuscript.pdf` | The same manuscript as a PDF |
| `manuscript.docx` | The same manuscript as a Word file |
| `highlights.txt` | Highlights source; the upload file is `Highlights.docx` |
| `Highlights.docx` | Highlights file for the Elsevier submission form |
| `Declaration-of-competing-interests.docx` | Separate competing-interests file the form requires |
| `fig1_results.png` | The one results image: plotted paired cost advantage of KG-EWS |
| `cohort_results.csv` | Primary cohorts and the predicate-flip cohorts |

## What the experiment claims

On 120 cohorts, KG-EWS has the lowest decision cost and the highest AUC among logistic regression, gradient boosting, the rules alone, and the combined score. The rules alone alarm on most surviving firms. The gain over logistic regression is small, paired, and absent from the cohorts that set the evidence weight to zero. It remains at a miss weight of 3, is not detectable at a miss weight of 8, and does not survive flipping each predicate with probability 0.15. The data are not a field sample and there is no failure date.

## Highlights

Knowledge-Based Systems asks for highlights as a separate editable file whose name contains “highlights.” Upload [Highlights.docx](Highlights.docx). The same five bullets are in [highlights.txt](highlights.txt). Each bullet is at most 85 characters.

- KG-EWS adds committee-rule evidence to a logistic warning score.
- Validation sets the evidence weight from a five-to-one miss cost.
- Across 120 cohorts the paired cost gain over logistic regression is 0.024.
- That gain remains at a three-to-one cost and is lost if predicates are flipped.
- Rules alone catch failures by alarming on most surviving firms.

## Elsevier submission

The new-submission form is [submit.elsevier.com/KNOSYS/submissions](https://submit.elsevier.com/KNOSYS/submissions). Sign in with an Elsevier account. These files are arranged so the form can read the manuscript and so the required uploads are present. Completing the submission still requires your login, your contact details, and your answers to the ethics and copyright questions. Editorial review begins after the form is submitted.

1. Choose the article type **Short communication**.
2. Upload `manuscript.docx` as the manuscript source. The title, the author name, the word Abstract, and the word Keywords are each on their own line so the form can extract them. `manuscript.pdf` is a preview.
3. Upload `Declaration-of-competing-interests.docx` when the form asks for the declaration of competing interests. The same statement is in the manuscript.
4. Upload `Highlights.docx` as the highlights file, and `fig1_results.png` if the form asks for the figure separately. The figure is also inside the manuscript.
5. For funding, choose no funding. The manuscript already contains the required sentence.
6. For the ethics question, this study uses synthetic cohorts and does not enroll human or animal subjects. Select the option the form gives for work that is not human or animal research, after you confirm that description.
7. For generative AI, the manuscript contains the required declaration. It names Cursor as the tool used to draft language and structure the code, and it states that the author reviewed the content.
8. Copy the CRediT roles from the manuscript into the form. There is one author.
9. On the title page and in the form, enter the affiliation, the full postal address, and the corresponding-author e-mail. Those three items are not in the files.
10. Mark Divya Munagala as the corresponding author, review the extracted title, abstract, and keywords, agree to the ethics and copyright questions, and complete the submission yourself.

After the confirmation message, register for Editorial Manager with the same e-mail address. Revisions are handled there.

The manuscript is set in Times New Roman on letter paper, double-spaced, with a centered title, the article type, and the author name. It is 12 pages, including the tables, the figure, and a double-spaced reference list. Knowledge-Based Systems asks a short communication to stay within 10 double-spaced pages, so the reference list can be tightened again before upload if the journal requires that cap.
