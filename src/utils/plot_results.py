#!/usr/bin/env python3
"""
FCBench veracity-prediction visualisation.

Produces 6 figures saved to experiment_results/figures/:
  fig1_macro_f1_heatmap.png       – Macro-F1 overview (model × dataset)
  fig2_per_class_f1.png           – Per-class F1 bars per dataset
  fig3_label_distribution.png     – Predicted vs gold label distributions
  fig4_seed_variance.png          – Macro-F1 variance across seeds (box plots)
  fig5_precision_recall.png       – Precision vs Recall scatter per class
  fig6_claim_agreement.png        – Per-claim model agreement histogram

Evaluation-split note
─────────────────────
  • tfidf/bm25/distilRoBERTa/Longformer : evaluated on a 20 % hold-out of
    train.pkl (reproducible with seed=42).  Aggregate metrics come from
    results.json; per-claim analysis reconstructs this split on the fly.
  • Sanctuary / AIC                      : evaluated on the dev split.
    Gold labels are derived from dev.pkl.
  Figures 1-5 compare both groups (noting the split difference).
  Figure 6 keeps the two groups separate to stay on the same claims.
"""

import json
import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split as _tts

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10})

# ── Paths ────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.join(BASE, "experiment_results")
DATA = os.path.join(BASE, "data", "cleaned_datasets")
OUT  = os.path.join(EXP, "figures")
os.makedirs(OUT, exist_ok=True)

DATASETS = ["averitec", "scifact", "climatecheck", "climatefever"]

# ── Display names / colours / label orders ───────────────────────────────────

MODEL_KEY_TO_DISPLAY = {
    "tfidf_logreg":   "TF-IDF+LR",
    "bm25_logreg":    "BM25+LR",
    "distilbert":     "distilRoBERTa",
    "longformer":     "Longformer",
    "sanctuary":      "Sanctuary",
    "aic":            "AIC",
    "llama_8b_none":  "Llama-8B",
    "llama_70b_none": "Llama-70B",
    "llama_8b_bm25":  "BM25+Llama-8B",
    "llama_70b_bm25": "BM25+Llama-70B",
}
ALL_MODELS = list(MODEL_KEY_TO_DISPLAY.values())

PALETTE = {
    "TF-IDF+LR":      "#4878d0",
    "BM25+LR":        "#ee854a",
    "distilRoBERTa":  "#6acc65",
    "Longformer":     "#d65f5f",
    "Sanctuary":      "#956cb4",
    "AIC":            "#8c613c",
    "Llama-8B":       "#4cb9e7",
    "Llama-70B":      "#1a7fa8",
    "BM25+Llama-8B":  "#e77cc1",
    "BM25+Llama-70B": "#a83d85",
}

# Consistent label ordering per dataset
LABEL_ORDER = {
    "averitec":    ["Supported", "Refuted", "Not Enough Evidence",
                    "Conflicting Evidence/Cherrypicking"],
    "scifact":     ["Supported", "Refuted", "Not Enough Evidence"],
    "climatecheck":["Supported", "Refuted", "Not Enough Evidence"],
    "climatefever":["Supported", "Refuted", "Not Enough Evidence",
                    "Conflicting Evidence/Cherrypicking"],
}

# Label normalisation (climatecheck prediction files use raw training label names)
CC_NORM = {
    "Supports":               "Supported",
    "Refutes":                "Refuted",
    "Not Enough Information": "Not Enough Evidence",
}
# Applied to every prediction label loaded from JSON files
PRED_LABEL_NORM = CC_NORM  # same mapping; extend here if other datasets diverge
CF_MAP = {
    0: "Supported",
    1: "Refuted",
    2: "Not Enough Evidence",
    3: "Conflicting Evidence/Cherrypicking",
}

# ── Gold-label helpers ───────────────────────────────────────────────────────

def _scifact_label(evidence: dict) -> str:
    if not evidence:
        return "Not Enough Evidence"
    labels = set()
    for doc in evidence.values():
        if doc is None:
            continue
        for e in doc:
            labels.add(e["label"])
    if "CONTRADICT" in labels:
        return "Refuted"
    if "SUPPORT" in labels:
        return "Supported"
    return "Not Enough Evidence"


def load_dev_gold(dataset: str) -> pd.DataFrame:
    """Return DataFrame(claim_id, gold_label) for the dev split."""
    dev = pd.read_parquet(os.path.join(DATA, dataset, "dev.pkl"))

    if dataset == "averitec":
        dev = dev.reset_index(drop=True)
        dev["claim_id"] = dev.index.astype(int)
        return dev[["claim_id", "label"]].rename(columns={"label": "gold_label"})

    if dataset == "scifact":
        dev["gold_label"] = dev["evidence"].apply(_scifact_label)
        return dev[["id", "gold_label"]].rename(columns={"id": "claim_id"})

    if dataset == "climatecheck":
        dev["gold_label"] = dev["label"].map(CC_NORM).fillna(dev["label"])
        return dev[["claim_id", "gold_label"]].astype({"claim_id": int})

    if dataset == "climatefever":
        dev = dev.drop_duplicates(subset="claim_id", keep="first").copy()
        dev["gold_label"] = dev["claim_label"].map(CF_MAP)
        return dev[["claim_id", "gold_label"]].astype({"claim_id": int})

    raise ValueError(dataset)


def load_dev_gold_seq(dataset: str) -> pd.DataFrame:
    """Return DataFrame(claim_id, gold_label) for the dev split with sequential ids.

    All prediction files — baseline and LLM — assign claim_id as the 0-based
    row index in dev.pkl (after dataset-specific preprocessing).  This function
    reproduces that same indexing so merges work correctly.
    """
    path = os.path.join(DATA, dataset, "dev.pkl")
    dev = pd.read_parquet(path)

    if dataset == "averitec":
        dev = dev.reset_index(drop=True)
        dev["claim_id"] = dev.index.astype(int)
        return dev[["claim_id", "label"]].rename(columns={"label": "gold_label"})

    if dataset == "scifact":
        dev["gold_label"] = dev["evidence"].apply(_scifact_label)
        dev = dev.reset_index(drop=True)
        dev["claim_id"] = dev.index.astype(int)
        return dev[["claim_id", "gold_label"]]

    if dataset == "climatecheck":
        dev = dev.dropna(subset=["label"]).copy()
        dev = dev[dev["label"] != ""].copy()
        dev["gold_label"] = dev["label"].map(CC_NORM).fillna(dev["label"])
        dev = dev.reset_index(drop=True)
        dev["claim_id"] = dev.index.astype(int)
        return dev[["claim_id", "gold_label"]]

    if dataset == "climatefever":
        dev = dev.drop_duplicates(subset="claim_id", keep="first").copy()
        dev["gold_label"] = dev["claim_label"].map(CF_MAP)
        dev = dev.reset_index(drop=True)
        dev["claim_id"] = dev.index.astype(int)
        return dev[["claim_id", "gold_label"]]

    raise ValueError(dataset)


# ── Prediction-file helpers ──────────────────────────────────────────────────

PRED_FILES_TEST = {               # evaluated on the 20% test split
    "tfidf_logreg": "tfidf___LogReg_seed42_veracity_prediction.json",
    "bm25_logreg":  "bm25___LogReg_seed42_veracity_prediction.json",
    "distilbert":   "distilRoBERTa_seed42_veracity_prediction.json",
    "longformer":   "longformer_seed42_veracity_prediction.json",
}
PRED_FILES_DEV = {                # evaluated on the dev split
    "sanctuary":      "sanctuary_seed42_veracity_prediction.json",
    "aic":            "aic_seed42_veracity_prediction.json",
    "llama_8b_none":  "llama-3.1-8b_none_veracity_prediction.json",
    "llama_70b_none": "llama-3.1-70b_none_veracity_prediction.json",
    "llama_8b_bm25":  "llama-3.1-8b_bm25_veracity_prediction.json",
    "llama_70b_bm25": "llama-3.1-70b_bm25_veracity_prediction.json",
}


def _load_pred_json(dataset: str, filename: str) -> pd.DataFrame | None:
    path = os.path.join(EXP, dataset, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "claim_id" not in df.columns and "index_id" in df.columns:
        df = df.rename(columns={"index_id": "claim_id"})
    df["claim_id"] = df["claim_id"].astype(int)
    # Normalise label names (e.g. climatecheck: "Supports" → "Supported")
    df["pred_label"] = df["pred_label"].map(
        lambda x: PRED_LABEL_NORM.get(x, x)
    )
    # Keep one prediction per claim (some files have one row per evidence item)
    df = df.drop_duplicates(subset="claim_id", keep="first")
    return df[["claim_id", "pred_label"]]


def _merged(dataset: str, pred_files: dict, gold_fn) -> pd.DataFrame:
    """Load gold labels and merge with all available prediction files."""
    try:
        gold = gold_fn(dataset)
    except Exception as e:
        print(f"  Warning: could not load gold for {dataset}: {e}")
        return pd.DataFrame()

    merged = gold.copy()
    for key, fname in pred_files.items():
        preds = _load_pred_json(dataset, fname)
        if preds is None:
            continue
        preds = preds.rename(columns={"pred_label": key})
        merged = merged.merge(preds, on="claim_id", how="inner")
    return merged


# ── Aggregate metrics builder ────────────────────────────────────────────────

def _report(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def _llm_metrics(dataset: str, model_key: str) -> dict | None:
    fname = PRED_FILES_DEV[model_key]
    preds = _load_pred_json(dataset, fname)
    if preds is None:
        return None
    gold = load_dev_gold_seq(dataset)
    df = gold.merge(preds, on="claim_id", how="inner")
    df = df[df["gold_label"].notna() & (df["gold_label"] != "")]
    return _report(df["gold_label"], df["pred_label"])


def build_metrics_df() -> pd.DataFrame:
    rows = []
    skip = {"seed", "model", "accuracy", "macro avg", "weighted avg",
            "n_labels", "n_epoch"}

    with open(os.path.join(EXP, "results.json")) as f:
        results = json.load(f)

    for dataset, entries in results.items():
        for e in entries:
            row = {
                "dataset":    dataset,
                "model":      MODEL_KEY_TO_DISPLAY[e["model"]],
                "seed":       e["seed"],
                "macro_f1":   e["macro avg"]["f1-score"],
                "macro_prec": e["macro avg"]["precision"],
                "macro_rec":  e["macro avg"]["recall"],
                "accuracy":   e["accuracy"],
                "split":      "dev",
            }
            for k, v in e.items():
                if k not in skip and isinstance(v, dict):
                    row[f"f1__{k}"]   = v.get("f1-score", np.nan)
                    row[f"prec__{k}"] = v.get("precision", np.nan)
                    row[f"rec__{k}"]  = v.get("recall", np.nan)
            rows.append(row)

    for dataset in DATASETS:
        for key in PRED_FILES_DEV:
            rep = _llm_metrics(dataset, key)
            if rep is None:
                continue
            row = {
                "dataset":    dataset,
                "model":      MODEL_KEY_TO_DISPLAY[key],
                "seed":       42,
                "macro_f1":   rep["macro avg"]["f1-score"],
                "macro_prec": rep["macro avg"]["precision"],
                "macro_rec":  rep["macro avg"]["recall"],
                "accuracy":   rep.get("accuracy", np.nan),
                "split":      "dev",
            }
            for k, v in rep.items():
                if k not in ("macro avg", "weighted avg", "accuracy") \
                        and isinstance(v, dict):
                    row[f"f1__{k}"]   = v.get("f1-score", np.nan)
                    row[f"prec__{k}"] = v.get("precision", np.nan)
                    row[f"rec__{k}"]  = v.get("recall", np.nan)
            rows.append(row)

    return pd.DataFrame(rows)


# ── Figure 1 – Macro-F1 heatmap ──────────────────────────────────────────────

def fig1_macro_f1_heatmap(df: pd.DataFrame):
    seed42 = df[df["seed"] == 42]
    pivot = (
        seed42.groupby(["model", "dataset"])["macro_f1"]
        .mean()
        .unstack("dataset")
        .reindex(ALL_MODELS)
        .reindex(columns=DATASETS)
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".3f",
        cmap="YlOrRd", vmin=0.0, vmax=1.0,
        linewidths=0.6, ax=ax, annot_kws={"size": 11},
    )
    ax.set_title(
        "Macro-F1 by Model and Dataset",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_macro_f1_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig1_macro_f1_heatmap.png")


# ── Figure 2 – Per-class F1 bars ─────────────────────────────────────────────

def fig2_per_class_f1(df: pd.DataFrame):
    seed42 = df[df["seed"] == 42]
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx // 2][idx % 2]
        sub  = seed42[seed42["dataset"] == dataset]
        lbls = [l for l in LABEL_ORDER[dataset] if f"f1__{l}" in sub.columns]
        x    = np.arange(len(lbls))
        models_here = [m for m in ALL_MODELS if m in sub["model"].values]
        n    = len(models_here)
        w    = 0.75 / n

        for i, model in enumerate(models_here):
            row  = sub[sub["model"] == model].iloc[0]
            vals = [row.get(f"f1__{l}", 0) or 0 for l in lbls]
            offs = (i - n / 2 + 0.5) * w
            ax.bar(x + offs, vals, w, label=model,
                   color=PALETTE[model], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [l.replace("Conflicting Evidence/Cherrypicking", "Conflicting\nEvidence")
              .replace("Not Enough Evidence", "Not Enough\nEvidence")
             for l in lbls],
            fontsize=9,
        )
        ax.set_title(dataset, fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.25)
        if idx % 2 == 0:
            ax.set_ylabel("F1-score", fontsize=11)

    handles = [mpatches.Patch(color=PALETTE[m], label=m)
               for m in ALL_MODELS if m in df["model"].values]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-class F1 by Model", fontsize=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_per_class_f1.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_per_class_f1.png")


# ── Figure 3 – Predicted label distribution vs gold ──────────────────────────

def fig3_label_distribution(test_merged: dict, dev_merged: dict):
    """Stacked-bar prediction distribution for each model vs the gold baseline."""
    n_label_colors = 4
    label_pal = sns.color_palette("Set2", n_label_colors)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx // 2][idx % 2]
        lbls = LABEL_ORDER[dataset]
        color_map = {l: label_pal[i % n_label_colors] for i, l in enumerate(lbls)}

        # Build ordered list: gold + test models + dev models
        groups = []

        t_df = test_merged.get(dataset)
        t_models = [k for k in PRED_FILES_TEST if t_df is not None and k in t_df.columns]
        d_df = dev_merged.get(dataset)
        d_models = [k for k in PRED_FILES_DEV if d_df is not None and k in d_df.columns]

        if t_df is not None and len(t_df):
            groups.append(("Gold\n(dev)", t_df["gold_label"]))
            for k in t_models:
                groups.append((MODEL_KEY_TO_DISPLAY[k], t_df[k]))
        if d_df is not None and len(d_df):
            if not groups:
                groups.append(("Gold\n(dev)", d_df["gold_label"]))
            for k in d_models:
                groups.append((MODEL_KEY_TO_DISPLAY[k], d_df[k]))

        if not groups:
            ax.set_visible(False)
            continue

        bar_labels, series = zip(*groups)
        x = np.arange(len(bar_labels))
        bottom = np.zeros(len(bar_labels))

        for lbl in lbls:
            vals = np.array([s.value_counts(normalize=True).get(lbl, 0)
                             for s in series])
            ax.bar(x, vals, bottom=bottom,
                   color=color_map[lbl], alpha=0.88)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(dataset, fontsize=13)
        if idx % 2 == 0:
            ax.set_ylabel("Fraction of predictions", fontsize=10)
        ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.25)

    # Shared legend using the superset of labels (averitec has all 4)
    handles = [mpatches.Patch(color=label_pal[i % n_label_colors], label=l)
               for i, l in enumerate(LABEL_ORDER["averitec"])]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Predicted Label Distribution vs Gold", fontsize=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_label_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_label_distribution.png")


# ── Figure 4 – Seed variance ──────────────────────────────────────────────────

def fig4_seed_variance(df: pd.DataFrame):
    seeded = ["TF-IDF+LR", "BM25+LR", "distilRoBERTa"]
    sub = df[df["model"].isin(seeded)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, dataset in zip(axes, DATASETS):
        ds = sub[sub["dataset"] == dataset]
        data   = [ds[ds["model"] == m]["macro_f1"].values for m in seeded]
        bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))
        for patch, m in zip(bp["boxes"], seeded):
            patch.set_facecolor(PALETTE[m])
            patch.set_alpha(0.72)

        # Overlay individual seed points
        for i, (vals, m) in enumerate(zip(data, seeded), start=1):
            jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color=PALETTE[m], s=30, zorder=5, alpha=0.9)

        ax.set_xticks(range(1, len(seeded) + 1))
        ax.set_xticklabels(seeded, rotation=18, ha="right", fontsize=9)
        ax.set_title(dataset, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        if dataset == "averitec":
            ax.set_ylabel("Macro-F1", fontsize=11)

    fig.suptitle("Macro-F1 Variance Across Seeds  (seeds 42 / 43 / 26)",
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_seed_variance.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig4_seed_variance.png")


# ── Figure 5 – Precision vs Recall scatter ────────────────────────────────────

def fig5_precision_recall(df: pd.DataFrame):
    seed42 = df[df["seed"] == 42]
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))

    for ax, dataset in zip(axes, DATASETS):
        sub  = seed42[seed42["dataset"] == dataset]
        lbls = LABEL_ORDER[dataset]

        for _, row in sub.iterrows():
            model = row["model"]
            for lbl in lbls:
                p = row.get(f"prec__{lbl}", np.nan)
                r = row.get(f"rec__{lbl}",  np.nan)
                if pd.isna(p) or pd.isna(r):
                    continue
                ax.scatter(r, p, color=PALETTE.get(model, "gray"),
                           s=65, alpha=0.82, zorder=3,
                           edgecolors="white", linewidths=0.5)
                short = (lbl[:3] if lbl != "Conflicting Evidence/Cherrypicking"
                         else "CE")
                ax.annotate(short, (r, p), fontsize=6, alpha=0.65,
                            textcoords="offset points", xytext=(3, 3))

        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.35, label="P=R")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Recall", fontsize=10)
        if dataset == "averitec":
            ax.set_ylabel("Precision", fontsize=10)
        ax.set_title(dataset, fontsize=12)
        ax.grid(alpha=0.2)

    handles = [mpatches.Patch(color=PALETTE[m], label=m)
               for m in ALL_MODELS if m in seed42["model"].values]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Precision vs Recall per Class per Model  (seed 42)\n"
                 "Labels: Sup=Supported  Ref=Refuted  Not=Not Enough Evidence  "
                 "CE=Conflicting Evidence",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_precision_recall.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_precision_recall.png")


# ── Figure 6 – Per-claim model agreement ────────────────────────────────────

def fig6_claim_agreement(test_merged: dict, dev_merged: dict):
    """2-row grid: top = claim-only models (test split), bottom = LLM (dev split)."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    cool = sns.color_palette("coolwarm_r", 7)

    for col, dataset in enumerate(DATASETS):
        for row_idx, (merged, group_label, pred_keys) in enumerate([
            (test_merged.get(dataset), "Claim-only models (dev split)",
             list(PRED_FILES_TEST.keys())),
            (dev_merged.get(dataset),  "LLM models (dev split)",
             list(PRED_FILES_DEV.keys())),
        ]):
            ax = axes[row_idx][col]
            if merged is None or len(merged) == 0:
                ax.set_visible(False)
                continue

            avail = [k for k in pred_keys if k in merged.columns]
            if not avail:
                ax.set_visible(False)
                continue

            correct_mat = pd.DataFrame({
                k: (merged[k] == merged["gold_label"]).astype(int)
                for k in avail
            })
            n_correct = correct_mat.sum(axis=1)
            n_models  = len(avail)
            counts    = n_correct.value_counts().reindex(
                range(n_models + 1), fill_value=0
            )

            bars = ax.bar(
                counts.index, counts.values,
                color=[cool[min(i, len(cool) - 1)] for i in counts.index],
                edgecolor="white", alpha=0.88,
            )
            ax.set_xticks(range(n_models + 1))
            ax.set_xlabel("# models correct", fontsize=9)
            if col == 0:
                ax.set_ylabel("# claims", fontsize=9)
            ax.set_title(
                f"{dataset}\n({len(merged)} claims, "
                f"models: {', '.join(MODEL_KEY_TO_DISPLAY.get(k, k) for k in avail)})",
                fontsize=8,
            )
            ax.grid(axis="y", alpha=0.3)

            total = len(merged)
            pct_all_wrong = (n_correct == 0).sum() / total
            pct_all_right = (n_correct == n_models).sum() / total
            ax.text(0.03, 0.96, f"All wrong: {pct_all_wrong:.0%}",
                    transform=ax.transAxes, fontsize=8,
                    va="top", color="#c0392b")
            ax.text(0.97, 0.96, f"All right: {pct_all_right:.0%}",
                    transform=ax.transAxes, fontsize=8,
                    va="top", ha="right", color="#27ae60")

        # Row labels on left
    for row_idx, lbl in enumerate(
        ["Claim-only models (dev split)", "LLM models (dev split)"]
    ):
        axes[row_idx][0].set_ylabel(
            f"{lbl}\n\n# claims", fontsize=9
        )

    fig.suptitle("Per-claim Model Agreement  "
                 "(how many models predicted the correct label per claim)",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_claim_agreement.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_claim_agreement.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building aggregate metrics …")
    df = build_metrics_df()
    models_found = sorted(df["model"].unique())
    print(f"  {len(df)} rows | models: {models_found}")

    print("Loading per-claim predictions (baseline, dev split) …")
    test_merged = {
        ds: _merged(ds, PRED_FILES_TEST, load_dev_gold_seq)
        for ds in DATASETS
    }
    for ds, m in test_merged.items():
        cols = [c for c in m.columns if c not in ("claim_id", "gold_label")]
        print(f"  {ds}: {len(m)} claims | {cols}")

    print("Loading per-claim predictions (LLM, dev split) …")
    dev_merged = {
        ds: _merged(ds, PRED_FILES_DEV, load_dev_gold_seq)
        for ds in DATASETS
    }
    for ds, m in dev_merged.items():
        cols = [c for c in m.columns if c not in ("claim_id", "gold_label")]
        print(f"  {ds}: {len(m)} claims | {cols}")

    print("\nGenerating figures …")
    fig1_macro_f1_heatmap(df)
    fig2_per_class_f1(df)
    fig3_label_distribution(test_merged, dev_merged)
    fig4_seed_variance(df)
    fig5_precision_recall(df)
    fig6_claim_agreement(test_merged, dev_merged)

    print(f"\nDone — all figures in {OUT}/")


if __name__ == "__main__":
    main()
