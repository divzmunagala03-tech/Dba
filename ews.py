"""KG-EWS: knowledge-guided early warning for digital-transformation failure.

A reproducible multi-cohort benchmark. Committee rules are coded on the
operational venture state. Statistical models see noisy measurements of
that state. The warning score adds rule evidence to a logistic log-odds.
The same file then re-selects the evidence weight at two other miss
costs and repeats the comparison after the predicates are flipped.
Every number in PAPER.md is written by this file.

Run: python ews.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SEED0 = 9000
N_COHORTS = 120
N_VENTURES = 380
MISS_WEIGHT = 5.0
FALSE_ALARM_WEIGHT = 1.0
SENSITIVITY_WEIGHTS = (3.0, 8.0)
STRESS_SEED0 = 12000
N_STRESS = 60
STRESS_FLIP = 0.15
LAMBDA_GRID = np.linspace(0.0, 3.0, 16)

# Expert evidence weights. These are not the data-generating coefficients.
RULES = [
    ("R1", 1.00, "Digital investment above 0.62 and capability below 0.38"),
    ("R2", 0.85, "Digital investment above 0.58 and innovation throughput below 0.28"),
    ("R3", 0.80, "Digital investment above 0.52 and team digital experience below 0.32"),
    ("R4", 0.95, "Runway under 5.5 months and burn multiple above 2.2"),
    ("R5", 0.60, "Customer concentration above 0.72 and market turbulence above 0.58"),
    ("R6", 0.55, "Three or more pivots and adoption lag above 0.65"),
    ("R7", 0.75, "Thin digital team, weak innovation, and material digital investment"),
    ("P1", -1.05, "Strong capability, strong innovation, and runway above 14 months"),
]
CF = np.array([rule[1] for rule in RULES])
FEATURE_NAMES = [
    "runway", "burn", "d_inv", "d_cap", "innov",
    "team", "turb", "conc", "pivots", "lag",
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def rule_matrix(clean: np.ndarray) -> np.ndarray:
    runway, burn, d_inv, d_cap, innov, team, turb, conc, pivots, lag = clean.T
    return np.column_stack([
        (d_inv > 0.62) & (d_cap < 0.38),
        (d_inv > 0.58) & (innov < 0.28),
        (d_inv > 0.52) & (team < 0.32),
        (runway < 5.5) & (burn > 2.2),
        (conc > 0.72) & (turb > 0.58),
        (pivots >= 3) & (lag > 0.65),
        (team < 0.28) & (innov < 0.32) & (d_inv > 0.48),
        (d_cap > 0.68) & (innov > 0.58) & (runway > 14),
    ]).astype(float)


def generate_cohort(rng: np.random.Generator, n: int = N_VENTURES):
    """Operational state, noisy measurements, and a documented failure draw."""
    runway = rng.uniform(1.0, 24.0, n)
    burn = np.clip(rng.lognormal(0.28, 0.28, n), 0.3, 6.0)
    d_inv = rng.beta(2.0, 2.4, n)
    d_cap = np.clip(rng.beta(2.4, 2.0, n), 0, 1)
    innov = rng.beta(2.2, 2.2, n)
    team = np.clip(rng.beta(2.2, 2.1, n), 0, 1)
    turb = rng.beta(1.6, 3.0, n)
    conc = rng.beta(1.9, 2.4, n)
    pivots = rng.poisson(0.9, n).astype(float)
    lag = rng.beta(2.0, 2.4, n)
    clean = np.column_stack([runway, burn, d_inv, d_cap, innov, team, turb, conc, pivots, lag])
    fired = rule_matrix(clean)
    risk = fired[:, :7].any(axis=1)
    eta = (
        -2.15
        + 0.16 * (11.0 - runway)
        + 0.85 * (burn - 1.2)
        + 1.35 * turb
        + 1.10 * conc
        + 1.70 * fired[:, 0] + 1.40 * fired[:, 1] + 1.20 * fired[:, 2] + 1.50 * fired[:, 3]
        + 1.00 * fired[:, 4] + 0.80 * fired[:, 5] + 1.15 * fired[:, 6] - 1.55 * fired[:, 7]
        + rng.normal(0.0, 0.35, n)
    )
    y = (rng.random(n) < sigmoid(eta)).astype(int)
    # Instrument noise. Rule-defining digital fields are noisier than the
    # financial and market fields a data model can measure directly.
    scale = np.array([0.80, 0.15, 0.18, 0.18, 0.18, 0.18, 0.04, 0.04, 0.70, 0.18])
    noisy = clean + rng.normal(0.0, 1.0, clean.shape) * scale
    noisy[:, 0] = np.clip(noisy[:, 0], 0.4, 30)
    noisy[:, 1] = np.clip(noisy[:, 1], 0.2, 8)
    for column in (2, 3, 4, 5, 6, 7, 9):
        noisy[:, column] = np.clip(noisy[:, column], 0, 1)
    noisy[:, 8] = np.clip(noisy[:, 8], 0, 8)
    return noisy, y, fired, risk


def objective_parts(y, score, risk, tau, miss_weight=MISS_WEIGHT):
    """Per-firm decision cost, plus miss rates inside each failure mode.

    Cost = (miss_weight * missed failures + 1 * false alarms) / n.
    Alerting every firm therefore costs about the non-failure rate, and a
    warning is useful only when it cuts missed failures by more than the
    false alarms it adds. The primary benchmark uses miss_weight 5.
    """
    alert = score >= tau
    rule_fail = (y == 1) & risk
    smooth_fail = (y == 1) & ~risk
    missed = (y == 1) & ~alert
    false_alarms = (y == 0) & alert

    def miss(group):
        if group.sum() == 0:
            return 0.0
        return float((~alert & group).sum() / group.sum())

    cost = float((miss_weight * missed.sum() + FALSE_ALARM_WEIGHT * false_alarms.sum()) / len(y))
    negatives = max(int((y == 0).sum()), 1)
    false_alarm_rate = float(false_alarms.sum() / negatives)
    justified = float((alert & risk).sum() / alert.sum()) if alert.any() else 0.0
    return cost, miss(rule_fail), miss(smooth_fail), false_alarm_rate, justified


def decision_objective(parts) -> float:
    return float(parts[0])


def best_threshold(y, score, risk, miss_weight=MISS_WEIGHT) -> float:
    grid = np.unique(np.quantile(score, np.linspace(0.05, 0.95, 41)))
    best_tau, best_j = float(np.median(score)), 1e18
    for tau in grid:
        value = decision_objective(objective_parts(y, score, risk, float(tau), miss_weight))
        if value < best_j:
            best_tau, best_j = float(tau), value
    return best_tau


def choose_lambda(p_lin, evidence, y, risk, miss_weight=MISS_WEIGHT) -> float:
    """Smallest grid weight that minimizes validation decision cost."""
    best = None
    for lam in LAMBDA_GRID:
        score = sigmoid(logit(p_lin) + lam * evidence)
        tau = best_threshold(y, score, risk, miss_weight)
        value = decision_objective(objective_parts(y, score, risk, tau, miss_weight))
        if best is None or value < best[0]:
            best = (value, float(lam))
    return best[1]


def _gbdt() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.90,
        min_samples_leaf=12,
        random_state=42,
    )


def evaluate_cohort(cohort_id: int, seed: int | None = None, flip_p: float = 0.0, with_sensitivity: bool = False) -> dict:
    seed = SEED0 + cohort_id if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    measured, y, fired, risk = generate_cohort(rng)
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"Cohort {cohort_id} has a single class.")
    # Stress path: each predicate is flipped before the evidence is summed.
    # The failure label and the clean risk flag are left as drawn.
    if flip_p > 0:
        fired = np.where(rng.random(fired.shape) < flip_p, 1.0 - fired, fired)
    evidence = fired @ CF
    index = np.arange(len(y))
    train, test = train_test_split(index, test_size=0.30, stratify=y, random_state=seed)
    fit, val = train_test_split(train, test_size=0.25, stratify=y[train], random_state=seed + 17)

    gbdt = _gbdt()
    gbdt.fit(measured[fit], y[fit])
    p_ai_val = gbdt.predict_proba(measured[val])[:, 1]
    p_ai_test = gbdt.predict_proba(measured[test])[:, 1]

    scaler = StandardScaler().fit(measured[fit])
    linear = LogisticRegression(max_iter=800, class_weight="balanced")
    linear.fit(scaler.transform(measured[fit]), y[fit])
    p_lin_val = linear.predict_proba(scaler.transform(measured[val]))[:, 1]
    p_lin_test = linear.predict_proba(scaler.transform(measured[test]))[:, 1]

    knowledge = LogisticRegression(max_iter=500)
    knowledge.fit(evidence[fit].reshape(-1, 1), y[fit])
    p_kb_val = knowledge.predict_proba(evidence[val].reshape(-1, 1))[:, 1]
    p_kb_test = knowledge.predict_proba(evidence[test].reshape(-1, 1))[:, 1]

    # The measured fields are mostly additive. Rule evidence carries the
    # conjunctions a linear score cannot represent, so the knowledge weight
    # is applied to the logistic log-odds rather than to the tree score.
    lam = choose_lambda(p_lin_val, evidence[val], y[val], risk[val], MISS_WEIGHT)
    kg_val = sigmoid(logit(p_lin_val) + lam * evidence[val])
    kg_test = sigmoid(logit(p_lin_test) + lam * evidence[test])

    val_scores = {"lin": p_lin_val, "kb": p_kb_val, "ai": p_ai_val, "kg": kg_val}
    test_scores = {"lin": p_lin_test, "kb": p_kb_test, "ai": p_ai_test, "kg": kg_test}
    record = {
        "cohort": cohort_id,
        "lambda": lam,
        "failure_rate": float(y.mean()),
        "rule_failure_share": float(((y == 1) & risk).sum() / max(int(y.sum()), 1)),
    }
    for name in test_scores:
        tau = best_threshold(y[val], val_scores[name], risk[val])
        cost, miss_rule, miss_smooth, false_alarm, justified = objective_parts(
            y[test], test_scores[name], risk[test], tau
        )
        record[f"{name}_auc"] = float(roc_auc_score(y[test], test_scores[name]))
        record[f"{name}_j"] = cost
        record[f"{name}_miss_rule"] = miss_rule
        record[f"{name}_miss_smooth"] = miss_smooth
        record[f"{name}_false_alarm"] = false_alarm
        record[f"{name}_justified"] = justified

    if with_sensitivity:
        for weight in SENSITIVITY_WEIGHTS:
            tag = f"w{int(weight)}"
            lam_w = choose_lambda(p_lin_val, evidence[val], y[val], risk[val], weight)
            kg_val_w = sigmoid(logit(p_lin_val) + lam_w * evidence[val])
            kg_test_w = sigmoid(logit(p_lin_test) + lam_w * evidence[test])
            tau_lin = best_threshold(y[val], p_lin_val, risk[val], weight)
            tau_kg = best_threshold(y[val], kg_val_w, risk[val], weight)
            record[f"lambda_{tag}"] = lam_w
            record[f"lin_j_{tag}"] = objective_parts(y[test], p_lin_test, risk[test], tau_lin, weight)[0]
            record[f"kg_j_{tag}"] = objective_parts(y[test], kg_test_w, risk[test], tau_kg, weight)[0]

    case = _representative_case(
        y[test], risk[test], fired[test], evidence[test],
        p_lin_test, p_kb_test, p_ai_test, kg_test, lam,
    )
    record["case"] = case
    return record


def _representative_case(y, risk, fired, evidence, p_lin, p_kb, p_ai, p_kg, lam) -> dict | None:
    if lam <= 0:
        return None
    lift = p_kg - p_lin
    candidates = np.where((y == 1) & risk & (fired[:, :7].sum(axis=1) >= 1) & (lift >= 0.08))[0]
    if len(candidates) == 0:
        return None
    pick = int(candidates[np.argmax(lift[candidates])])
    fired_rules = []
    for index, (rule_id, weight, text) in enumerate(RULES):
        if fired[pick, index] == 1:
            fired_rules.append({"id": rule_id, "weight": weight, "text": text, "evidence": float(weight)})
    return {
        "failed": True,
        "rule_triggered": bool(risk[pick]),
        "evidence": float(evidence[pick]),
        "p_lin": float(p_lin[pick]),
        "p_kb": float(p_kb[pick]),
        "p_ai": float(p_ai[pick]),
        "p_kg": float(p_kg[pick]),
        "rules": fired_rules,
    }


def summarize(frame: pd.DataFrame) -> dict:
    methods = ("lin", "kb", "ai", "kg")
    metric_keys = {
        "auc": "_auc",
        "j": "_j",
        "miss_rule": "_miss_rule",
        "miss_smooth": "_miss_smooth",
        "false_alarm": "_false_alarm",
        "justified": "_justified",
    }
    summary = {"n_cohorts": int(len(frame))}
    for method in methods:
        summary[method] = {}
        for label, suffix in metric_keys.items():
            series = frame[f"{method}{suffix}"].to_numpy()
            mean = float(series.mean())
            sd = float(series.std(ddof=1))
            half = float(stats.t.ppf(0.975, len(series) - 1) * sd / np.sqrt(len(series)))
            summary[method][label] = {"mean": mean, "std": sd, "ci95": [mean - half, mean + half]}
    tests = []
    for method, label in (("lin", "Logistic regression"), ("ai", "Gradient boosting"), ("kb", "Knowledge base")):
        for metric, suffix, expect_kg_lower in (
            ("objective", "_j", True),
            ("auc", "_auc", False),
        ):
            kg = frame[f"kg{suffix}"].to_numpy()
            base = frame[f"{method}{suffix}"].to_numpy()
            if expect_kg_lower:
                result = stats.wilcoxon(base, kg, alternative="greater", zero_method="wilcox")
                diff = base - kg
            else:
                result = stats.wilcoxon(kg, base, alternative="greater", zero_method="wilcox")
                diff = kg - base
            mean = float(diff.mean())
            sd = float(diff.std(ddof=1))
            half = float(stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff)))
            better = diff[diff > 0]
            worse = diff[diff < 0]
            tests.append({
                "comparison": f"KG-EWS vs {label}",
                "metric": metric,
                "kg_mean": float(kg.mean()),
                "baseline_mean": float(base.mean()),
                "mean_advantage": mean,
                "ci95": [mean - half, mean + half],
                "n_better": int((diff > 0).sum()),
                "n_worse": int((diff < 0).sum()),
                "n_tie": int((diff == 0).sum()),
                "mean_when_better": float(better.mean()) if len(better) else None,
                "mean_when_worse": float(worse.mean()) if len(worse) else None,
                "p_value": float(result.pvalue),
            })
    summary["tests"] = tests
    summary["lambda_mean"] = float(frame["lambda"].mean())
    summary["lambda_std"] = float(frame["lambda"].std(ddof=1))
    summary["failure_rate"] = float(frame["failure_rate"].mean())
    summary["rule_failure_share"] = float(frame["rule_failure_share"].mean())
    summary["share_lambda_positive"] = float((frame["lambda"] > 0).mean())
    return summary


def plot_results(path: Path, summary: dict, cohorts: pd.DataFrame | None = None) -> None:
    """One results image: cohort-level paired cost advantage, plotted."""
    if cohorts is None:
        cohorts = pd.read_csv(ROOT / "cohort_results.csv")
    if "block" in cohorts.columns:
        cohorts = cohorts.loc[cohorts["block"] == "main"].copy()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=300)
    fig.patch.set_facecolor("white")

    order = ("Logistic regression", "Knowledge base", "Gradient boosting")
    columns = ("lin_j", "kb_j", "ai_j")
    by_name = {row["comparison"]: row for row in summary["tests"] if row["metric"] == "objective"}
    rows = [by_name[f"KG-EWS vs {name}"] for name in order]
    colors = ["#78716c", "#0f766e", "#1d4ed8"]
    tick_labels = ["Versus\nlogistic regression", "Versus\nknowledge base", "Versus\ngradient boosting"]
    rng = np.random.default_rng(0)

    ax.axhline(0.0, color="#1c1917", linewidth=0.9, zorder=1)
    plotted = []
    for i, (column, row, color) in enumerate(zip(columns, rows, colors)):
        values = cohorts[column].to_numpy() - cohorts["kg_j"].to_numpy()
        plotted.append(values)
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.plot(
            i + jitter, values, linestyle="none", marker="o",
            markersize=3.4, color=color, alpha=0.38, markeredgewidth=0, zorder=2,
        )
        mean = row["mean_advantage"]
        low = mean - row["ci95"][0]
        high = row["ci95"][1] - mean
        ax.errorbar(
            i, mean, yerr=[[low], [high]], fmt="o",
            color="#1c1917", ecolor="#1c1917", elinewidth=1.15, capsize=4, markersize=8,
            markerfacecolor=color, markeredgecolor="#1c1917", markeredgewidth=0.8, zorder=4,
        )
        ax.annotate(
            f"{mean:.3f}", (i, mean + high), textcoords="offset points", xytext=(0, 6),
            ha="center", va="bottom", fontsize=9, color="#1c1917",
        )

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("Paired cost reduction per firm\n(baseline − KG-EWS)")
    ax.set_title("Paired cost advantage of KG-EWS", loc="left", fontsize=12, color="#1c1917")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e7e5e4", zorder=0)
    ax.set_axisbelow(True)
    flat = np.concatenate(plotted)
    floor = min(0.0, float(flat.min()))
    ceiling = max(float(flat.max()), max(row["ci95"][1] for row in rows))
    pad = 0.08 * max(ceiling - floor, 0.05)
    ax.set_ylim(floor - pad, ceiling + 2.4 * pad)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sensitivity(frame: pd.DataFrame) -> dict:
    """Re-selected λ and τ at miss weights other than the primary 5:1 cost."""
    out = {}
    for weight in SENSITIVITY_WEIGHTS:
        tag = f"w{int(weight)}"
        diff = frame[f"lin_j_{tag}"].to_numpy() - frame[f"kg_j_{tag}"].to_numpy()
        result = stats.wilcoxon(frame[f"lin_j_{tag}"], frame[f"kg_j_{tag}"], alternative="greater", zero_method="wilcox")
        mean = float(diff.mean())
        sd = float(diff.std(ddof=1))
        half = float(stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff)))
        out[tag] = {
            "miss_weight": weight,
            "lambda_mean": float(frame[f"lambda_{tag}"].mean()),
            "lambda_positive": float((frame[f"lambda_{tag}"] > 0).mean()),
            "lin_cost": float(frame[f"lin_j_{tag}"].mean()),
            "kg_cost": float(frame[f"kg_j_{tag}"].mean()),
            "mean_advantage": mean,
            "ci95": [mean - half, mean + half],
            "n_better": int((diff > 0).sum()),
            "n_worse": int((diff < 0).sum()),
            "n_tie": int((diff == 0).sum()),
            "p_value": float(result.pvalue),
        }
    return out


def main() -> None:
    records = []
    cases = []
    for cohort_id in range(N_COHORTS):
        record = evaluate_cohort(cohort_id, with_sensitivity=True)
        cases.append(record.pop("case"))
        record["block"] = "main"
        records.append(record)
    cases = [case for case in cases if case is not None]
    case = max(cases, key=lambda item: item["p_kg"] - item["p_lin"])

    stress_records = []
    for cohort_id in range(N_STRESS):
        record = evaluate_cohort(
            cohort_id,
            seed=STRESS_SEED0 + cohort_id,
            flip_p=STRESS_FLIP,
            with_sensitivity=False,
        )
        record.pop("case", None)
        record["block"] = "stress"
        stress_records.append(record)

    frame = pd.DataFrame(records)
    stress = pd.DataFrame(stress_records)
    pd.concat([frame, stress], ignore_index=True).to_csv(ROOT / "cohort_results.csv", index=False)
    summary = summarize(frame)
    summary["sensitivity"] = _sensitivity(frame)
    summary["stress"] = summarize(stress)
    summary["stress"]["n_cohorts"] = int(len(stress))
    summary["stress"]["flip_p"] = STRESS_FLIP
    summary["stress"]["seed0"] = STRESS_SEED0
    summary["case"] = case
    summary["n_ventures"] = N_VENTURES
    summary["miss_weight"] = MISS_WEIGHT
    summary["false_alarm_weight"] = FALSE_ALARM_WEIGHT
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(ROOT / "fig1_results.png", summary, frame)
    printable = {key: summary[key] for key in summary if key != "case"}
    printable["case"] = case
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
