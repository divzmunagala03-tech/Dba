"""Knowledge-based early-warning system for digital-venture failure.

The knowledge base is nine antecedent rules. Their weights are the
coefficients of a logistic model fit on the training fold only.
A gradient-boosting model scores the same ventures from the raw
features. A convex mixture of the two probabilities is estimated on
inner validation data and reported as a comparator.

Running this file rewrites the two figures and results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "venture_data.csv"
SEED = 42

# Expert certainty factors are a prior sign, not the weights used to score.
# P1 is protective, so its expert factor is negative.
RULES = [
    ("R1", 0.74, "No data-science skill and no unstructured-data proficiency"),
    ("R2", 0.70, "Cloud or platform offer, no team growth, and at most one senior leader"),
    ("R3", 0.72, "Capital-intensive firm with no repeat investor and no top-tier fund"),
    ("R4", 0.68, "Founders lack relevant experience and successful-startup experience"),
    ("R5", 0.66, "Online firm with non-positive internet activity and no team growth"),
    ("R6", 0.63, "Global and capital-intensive firm with at most one senior leader"),
    ("R7", 0.76, "Machine-learning or predictive-analytics offer with no data-science skill"),
    ("R8", 0.60, "Highly disruptive technology, slow time to market, and no entry barrier"),
    ("P1", -0.58, "Growing team, repeat capital, relevant experience, and data-science skill"),
]
RULE_IDS = [rule_id for rule_id, _, _ in RULES]

FEATURES = [
    "age", "internet", "employees", "log_funding", "n_seed", "n_angel",
    "n_cofounders", "n_advisors", "senior", "team_size", "repeat_investors",
    "months_to_first_funding", "industry_trend", "skill_ds", "skill_eng",
    "skill_ent", "skill_lead", "team_grown", "top_fund", "successful_startup",
    "capital_intensive", "ml", "predictive", "bigdata", "prescriptive", "speech",
    "cloud", "subscription", "global_player", "online", "b2b", "exp_ord",
    "edu_relevant", "exp_relevant", "tier1", "selling_ord", "team_comp_ord",
    "ttm_ord", "disrupt_ord", "tech_unstructured", "patent", "barriers",
    "gartner_ord", "research_ord",
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _yes(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[text.eq("yes")] = 1.0
    out.loc[text.eq("no")] = 0.0
    return out


def _ord(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(mapping).astype(float)


def load_ventures(path: Path = DATA) -> pd.DataFrame:
    """Digital ventures only. Company names are not in the file."""
    raw = pd.read_csv(path)
    frame = pd.DataFrame(index=raw.index)
    frame["y"] = raw["status"].astype(str).str.strip().eq("Failed").astype(int)
    frame["age"] = _num(raw["age"])
    frame["internet"] = _num(raw["internet"])
    frame["employees"] = _num(raw["employees"])
    funding = _num(raw["funding"]).clip(lower=0)
    frame["log_funding"] = np.log1p(funding)
    for source, target in (
        ("n_seed", "n_seed"),
        ("n_angel", "n_angel"),
        ("n_cofounders", "n_cofounders"),
        ("n_advisors", "n_advisors"),
        ("senior", "senior"),
        ("team_size", "team_size"),
        ("repeat_investors", "repeat_investors"),
        ("months_to_first_funding", "months_to_first_funding"),
        ("industry_trend", "industry_trend"),
        ("skill_ds", "skill_ds"),
        ("skill_eng", "skill_eng"),
        ("skill_ent", "skill_ent"),
        ("skill_lead", "skill_lead"),
    ):
        frame[target] = _num(raw[source])
    for source, target in (
        ("team_grown", "team_grown"),
        ("top_fund", "top_fund"),
        ("successful_startup", "successful_startup"),
        ("capital_intensive", "capital_intensive"),
        ("ml", "ml"),
        ("predictive", "predictive"),
        ("bigdata", "bigdata"),
        ("prescriptive", "prescriptive"),
        ("speech", "speech"),
        ("subscription", "subscription"),
        ("edu_relevant", "edu_relevant"),
        ("exp_relevant", "exp_relevant"),
        ("tech_unstructured", "tech_unstructured"),
        ("patent", "patent"),
        ("barriers", "barriers"),
    ):
        frame[target] = _yes(raw[source])

    cloud = raw["cloud"].astype(str).str.strip().str.lower()
    frame["cloud"] = np.where(cloud.isin(["cloud", "platform", "both"]), 1.0, np.where(cloud.eq("none"), 0.0, np.nan))
    reach = raw["reach"].astype(str).str.strip().str.lower()
    frame["global_player"] = np.where(reach.eq("global"), 1.0, np.where(reach.eq("local"), 0.0, np.nan))
    channel = raw["channel"].astype(str).str.strip().str.lower()
    frame["online"] = np.where(channel.isin(["online", "both"]), 1.0, np.where(channel.eq("offline"), 0.0, np.nan))
    market = raw["market"].astype(str).str.strip().str.lower()
    frame["b2b"] = np.where(market.eq("b2b"), 1.0, np.where(market.eq("b2c"), 0.0, np.nan))
    university = raw["university"].astype(str).str.strip().str.lower()
    frame["tier1"] = np.where(university.eq("tier_1"), 1.0, np.where(university.isin(["tier_2", "none"]), 0.0, np.nan))

    level = {"low": 0.0, "medium": 1.0, "high": 2.0}
    frame["exp_ord"] = _ord(raw["experience"], level)
    frame["selling_ord"] = _ord(raw["selling"], level)
    frame["team_comp_ord"] = _ord(raw["team_comp"], level)
    frame["ttm_ord"] = _ord(raw["ttm"], level)
    frame["disrupt_ord"] = _ord(raw["disrupt"], level)
    frame["research_ord"] = _ord(raw["research_pubs"], {"none": 0.0, "few": 1.0, "many": 2.0})
    frame["gartner_ord"] = _ord(raw["gartner"], {"trigger": 0.0, "peak": 1.0, "trough": 2.0, "slope": 3.0, "plateau": 4.0})

    digital = (
        frame["ml"].eq(1)
        | frame["predictive"].eq(1)
        | frame["bigdata"].eq(1)
        | frame["prescriptive"].eq(1)
        | frame["speech"].eq(1)
        | frame["cloud"].eq(1)
    )
    return frame.loc[digital].reset_index(drop=True)


def rule_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Fire a rule only when every antecedent is observed and holds."""
    senior_small = frame["senior"].le(1)
    no_ds = frame["skill_ds"].eq(0)
    fires = {
        "R1": no_ds & frame["tech_unstructured"].eq(0),
        "R2": frame["cloud"].eq(1) & frame["team_grown"].eq(0) & senior_small,
        "R3": frame["capital_intensive"].eq(1) & frame["repeat_investors"].eq(0) & frame["top_fund"].eq(0),
        "R4": frame["exp_relevant"].eq(0) & frame["successful_startup"].eq(0),
        "R5": frame["online"].eq(1) & frame["internet"].le(0) & frame["team_grown"].eq(0),
        "R6": frame["global_player"].eq(1) & frame["capital_intensive"].eq(1) & senior_small,
        "R7": (frame["ml"].eq(1) | frame["predictive"].eq(1)) & no_ds,
        "R8": frame["disrupt_ord"].eq(2) & frame["ttm_ord"].eq(2) & frame["barriers"].eq(0),
        "P1": (
            frame["team_grown"].eq(1)
            & frame["repeat_investors"].ge(1)
            & frame["exp_relevant"].eq(1)
            & frame["skill_ds"].gt(0)
        ),
    }
    return pd.DataFrame({rule_id: fires[rule_id] for rule_id in RULE_IDS}).astype(float)


def _gbdt() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.85,
        min_samples_leaf=8,
        random_state=SEED,
    )


def _logit() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED)


def _best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    grid = np.unique(np.quantile(score, np.linspace(0.05, 0.95, 37)))
    best_tau, best_f1 = float(np.median(score)), -1.0
    for tau in grid:
        value = f1_score(y, score >= tau, zero_division=0)
        if value > best_f1:
            best_tau, best_f1 = float(tau), float(value)
    return best_tau


def _best_alpha(y: np.ndarray, p_ai: np.ndarray, p_kb: np.ndarray) -> float:
    best_alpha, best_auc = 1.0, -1.0
    for alpha in np.linspace(0.0, 1.0, 21):
        if np.unique(y).size < 2:
            continue
        auc = roc_auc_score(y, alpha * p_ai + (1.0 - alpha) * p_kb)
        if auc > best_auc:
            best_alpha, best_auc = float(alpha), float(auc)
    return best_alpha


def _point_metrics(y: np.ndarray, score: np.ndarray, tau: float) -> dict[str, float]:
    alert = score >= tau
    return {
        "auc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, alert, zero_division=0)),
        "precision": float(precision_score(y, alert, zero_division=0)),
        "recall": float(recall_score(y, alert, zero_division=0)),
        "brier": float(brier_score_loss(y, np.clip(score, 0.0, 1.0))),
    }


def _fit_predict_ai(x_train, y_train, x_score) -> np.ndarray:
    model = _gbdt()
    model.fit(x_train, y_train)
    return model.predict_proba(x_score)[:, 1]


def _fit_predict_logit(x_train, y_train, x_score) -> np.ndarray:
    scaler = StandardScaler()
    model = _logit()
    model.fit(scaler.fit_transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_score))[:, 1]


def _fit_rule_model(fired_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = _logit()
    model.fit(fired_train, y_train)
    return model


def cross_validate(frame: pd.DataFrame) -> dict:
    y_all = frame["y"].to_numpy()
    fired = rule_matrix(frame).to_numpy(dtype=float)
    design = frame[FEATURES].to_numpy(dtype=float)
    names = ("kb", "ai", "logit", "fusion")
    oof = {name: np.zeros(len(frame), dtype=float) for name in names}
    fold_rows = []
    alphas = []
    coefs = []

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, y_all), start=1):
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + fold)
        fit_pos, val_pos = next(inner.split(train_idx, y_all[train_idx]))
        fit_idx = train_idx[fit_pos]
        val_idx = train_idx[val_pos]

        rule_fit = _fit_rule_model(fired[fit_idx], y_all[fit_idx])
        p_kb_val = rule_fit.predict_proba(fired[val_idx])[:, 1]
        imputer = SimpleImputer(strategy="median").fit(design[fit_idx])
        p_ai_val = _fit_predict_ai(imputer.transform(design[fit_idx]), y_all[fit_idx], imputer.transform(design[val_idx]))
        p_logit_val = _fit_predict_logit(imputer.transform(design[fit_idx]), y_all[fit_idx], imputer.transform(design[val_idx]))
        alpha = _best_alpha(y_all[val_idx], p_ai_val, p_kb_val)
        taus = {
            "kb": _best_threshold(y_all[val_idx], p_kb_val),
            "ai": _best_threshold(y_all[val_idx], p_ai_val),
            "logit": _best_threshold(y_all[val_idx], p_logit_val),
            "fusion": _best_threshold(y_all[val_idx], alpha * p_ai_val + (1.0 - alpha) * p_kb_val),
        }

        rule_model = _fit_rule_model(fired[train_idx], y_all[train_idx])
        coefs.append(rule_model.coef_.ravel().astype(float))
        imputer = SimpleImputer(strategy="median").fit(design[train_idx])
        x_train = imputer.transform(design[train_idx])
        x_test = imputer.transform(design[test_idx])
        p_kb = rule_model.predict_proba(fired[test_idx])[:, 1]
        p_ai = _fit_predict_ai(x_train, y_all[train_idx], x_test)
        p_logit = _fit_predict_logit(x_train, y_all[train_idx], x_test)
        p_fusion = alpha * p_ai + (1.0 - alpha) * p_kb
        scores = {"kb": p_kb, "ai": p_ai, "logit": p_logit, "fusion": p_fusion}
        entry = {"fold": fold, "alpha": alpha, "n_test": int(len(test_idx))}
        for name in names:
            oof[name][test_idx] = scores[name]
            entry[name] = _point_metrics(y_all[test_idx], scores[name], taus[name])
        fold_rows.append(entry)
        alphas.append(alpha)

    summary = {}
    for name in names:
        summary[name] = {
            metric: {
                "mean": float(np.mean([row[name][metric] for row in fold_rows])),
                "std": float(np.std([row[name][metric] for row in fold_rows], ddof=1)),
            }
            for metric in ("auc", "ap", "f1", "precision", "recall", "brier")
        }
        summary[name]["oof_auc"] = float(roc_auc_score(y_all, oof[name]))
        summary[name]["oof_ap"] = float(average_precision_score(y_all, oof[name]))
        summary[name]["oof_auc_ci"] = _bootstrap_auc(y_all, oof[name])

    coef_mat = np.vstack(coefs)
    return {
        "n": int(len(frame)),
        "n_source": 472,
        "failures": int(y_all.sum()),
        "failure_rate": float(y_all.mean()),
        "n_features": len(FEATURES),
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas, ddof=1)),
        "alphas": [float(a) for a in alphas],
        "summary": summary,
        "fusion_minus_ai": _bootstrap_diff(y_all, oof["fusion"], oof["ai"]),
        "folds": fold_rows,
        "rules": _rule_table(fired, y_all, coef_mat.mean(axis=0), coef_mat.std(axis=0, ddof=1)),
        "example": _example_case(frame, fired, y_all, oof, coef_mat.mean(axis=0)),
        "oof": {"y": y_all, **oof},
    }


def _bootstrap_auc(y: np.ndarray, score: np.ndarray, draws: int = 2000) -> list[float]:
    rng = np.random.default_rng(SEED)
    stats = []
    n = len(y)
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        stats.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return [float(lo), float(hi)]


def _bootstrap_diff(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, draws: int = 2000) -> dict:
    rng = np.random.default_rng(SEED + 1)
    stats = []
    n = len(y)
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        stats.append(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
    arr = np.asarray(stats)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {"mean": float(arr.mean()), "ci": [float(lo), float(hi)], "share_positive": float(np.mean(arr > 0))}


def _rule_table(fired: np.ndarray, y: np.ndarray, coef_mean: np.ndarray, coef_std: np.ndarray) -> list[dict]:
    rows = []
    for index, (rule_id, expert_cf, text) in enumerate(RULES):
        mask = fired[:, index] == 1.0
        rows.append({
            "id": rule_id,
            "expert_cf": expert_cf,
            "text": text,
            "coef_mean": float(coef_mean[index]),
            "coef_std": float(coef_std[index]),
            "support": float(mask.mean()),
            "n_fired": int(mask.sum()),
            "failure_rate": float(y[mask].mean()) if mask.any() else None,
        })
    return rows


def _example_case(frame, fired, y, oof, coef_mean) -> dict:
    risk_ids = [RULE_IDS[i] for i, coef in enumerate(coef_mean) if coef > 0]
    risk_index = [RULE_IDS.index(rule_id) for rule_id in risk_ids]
    n_risk = fired[:, risk_index].sum(axis=1) if risk_index else np.zeros(len(y))
    candidates = np.where((y == 1) & (n_risk >= 2) & (oof["ai"] >= 0.8))[0]
    if len(candidates) == 0:
        candidates = np.where(y == 1)[0]
    pick = int(candidates[np.argmax(oof["ai"][candidates])])
    fired_rows = []
    for index, rule_id in enumerate(RULE_IDS):
        if fired[pick, index] == 1.0:
            fired_rows.append({"id": rule_id, "coef": float(coef_mean[index])})
    return {
        "row": pick,
        "y": int(y[pick]),
        "p_kb": float(oof["kb"][pick]),
        "p_ai": float(oof["ai"][pick]),
        "p_fusion": float(oof["fusion"][pick]),
        "rules": fired_rows,
    }


def plot_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, face, edge):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            facecolor=face, edgecolor=edge, linewidth=1.3,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="#1c1917", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#44403c"))

    box(0.18, 0.84, 0.64, 0.12, "Venture data\nProfile of a digital entrepreneurial firm", "#f5f5f4", "#44403c")
    box(0.05, 0.50, 0.42, 0.24, "Knowledge base\nNine antecedent rules\nLogistic weights fit on the training fold\nOutput: which rules fired, and their sign", "#ccfbf1", "#0f766e")
    box(0.53, 0.50, 0.42, 0.24, "Artificial intelligence\nGradient boosting on 44 features\nMedian imputation inside the training fold\nOutput: failure probability", "#dbeafe", "#1d4ed8")
    box(0.16, 0.22, 0.68, 0.18, "Early-warning card\nProbability from the data model\nReasons from the fired rules\nA probability mixture is estimated, then kept only if it improves validation AUC", "#ffedd5", "#c2410c")
    box(0.22, 0.04, 0.56, 0.11, "Decision maker\nFounder, investor, or accelerator", "#ffe4e6", "#be123c")
    arrow(0.50, 0.84, 0.26, 0.74)
    arrow(0.50, 0.84, 0.74, 0.74)
    arrow(0.26, 0.50, 0.42, 0.40)
    arrow(0.74, 0.50, 0.58, 0.40)
    arrow(0.50, 0.22, 0.50, 0.15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_roc(path: Path, y: np.ndarray, scores: dict[str, np.ndarray], aucs: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    styles = [
        ("kb", "#0f766e", "Knowledge base"),
        ("logit", "#57534e", "Logistic regression"),
        ("ai", "#1d4ed8", "Gradient boosting"),
        ("fusion", "#c2410c", "Probability mixture"),
    ]
    for key, color, label in styles:
        fpr, tpr, _ = roc_curve(y, scores[key])
        ax.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{label}  {aucs[key]:.3f}")
    ax.plot([0, 1], [0, 1], color="#a8a29e", linewidth=1.0, linestyle="--")
    ax.set_xlabel("False-alarm rate")
    ax.set_ylabel("Detected failure rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=8.4, title="Out-of-fold AUC", title_fontsize=8.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    frame = load_ventures()
    if len(frame) < 400 or frame["y"].nunique() != 2:
        raise RuntimeError("Venture table did not produce the expected digital-firm cohort.")
    result = cross_validate(frame)
    plot_architecture(ROOT / "fig1_architecture.png")
    plot_roc(
        ROOT / "fig2_roc.png",
        result["oof"]["y"],
        {name: result["oof"][name] for name in ("kb", "logit", "ai", "fusion")},
        {name: result["summary"][name]["oof_auc"] for name in ("kb", "logit", "ai", "fusion")},
    )
    payload = {key: value for key, value in result.items() if key != "oof"}
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "n": payload["n"],
        "failures": payload["failures"],
        "alpha_mean": payload["alpha_mean"],
        "alphas": payload["alphas"],
        "oof_auc": {name: payload["summary"][name]["oof_auc"] for name in payload["summary"]},
        "oof_ap": {name: payload["summary"][name]["oof_ap"] for name in payload["summary"]},
        "ci": {name: payload["summary"][name]["oof_auc_ci"] for name in payload["summary"]},
        "fold_mean_auc": {name: payload["summary"][name]["auc"] for name in payload["summary"]},
        "fold_mean_f1": {name: payload["summary"][name]["f1"] for name in payload["summary"]},
        "fold_mean_precision": {name: payload["summary"][name]["precision"] for name in payload["summary"]},
        "fold_mean_recall": {name: payload["summary"][name]["recall"] for name in payload["summary"]},
        "fold_mean_brier": {name: payload["summary"][name]["brier"] for name in payload["summary"]},
        "fusion_minus_ai": payload["fusion_minus_ai"],
        "rules": payload["rules"],
        "example": payload["example"],
    }, indent=2))


if __name__ == "__main__":
    main()
