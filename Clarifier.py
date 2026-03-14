# =============================================================================
# EPOCH 3.0 DATATHON — PES University AI & ML
# Script  : classifier_train_final.py
# Role    : Binary Classifier (Book Now vs Wait)
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import os
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION & PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "df_classifier.csv")
MODEL_PATH = os.path.join(BASE_DIR, "clf_model_final.pkl")
PLOT_PATH = os.path.join(BASE_DIR, "classifier_report_final.png")
PLOT_COMPARE_PATH = os.path.join(BASE_DIR, "classifier_split_comparison.png")
PLOT_CURVES_PATH = os.path.join(BASE_DIR, "classifier_roc_pr_comparison.png")

# Aesthetic Colors
DARK, CARD, CYAN, GREEN, GOLD, TEXT = "#0f1117", "#1a1d27", "#00d4ff", "#00ff9d", "#ffd700", "#e0e0e0"

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  EPOCH 3.0 | Smart Booking Advisor — FINAL TRAINING")
print("=" * 62)

if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: 'df_classifier.csv' not found in {BASE_DIR}")
    exit()

df = pd.read_csv(DATA_PATH)
df = df[(df["days_left"] >= 1) & (df["days_left"] <= 49)].copy()
df.reset_index(drop=True, inplace=True)

TARGET = "is_good_price"

# ─────────────────────────────────────────────────────────────────────────────
# 2. THE "CORRECTION": REMOVING DATA LEAKAGE
# ─────────────────────────────────────────────────────────────────────────────
# We exclude 'price_ratio' because it's derived directly from the target.
# Using it results in 100% accuracy but fails on real-world unseen data.
FEATURES = [col for col in df.columns if col not in [TARGET, "price_ratio"]]

X = df[FEATURES]
y = df[TARGET]

print(f"✅ Data Loaded: {df.shape[0]:,} rows")
print(f"🛠️  Correction Applied: Removing 'price_ratio' to prevent leakage.")
print(f"📈 Features being used: {', '.join(FEATURES)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT (RANDOM + GROUPED)
# ─────────────────────────────────────────────────────────────────────────────
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

groups = df["route_enc"].astype(str) + "_" + df["class_enc"].astype(str)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
g_train_idx, g_test_idx = next(gss.split(X, y, groups=groups))

Xg_train, Xg_test = X.iloc[g_train_idx], X.iloc[g_test_idx]
yg_train, yg_test = y.iloc[g_train_idx], y.iloc[g_test_idx]

print(f"🧪 Random Split Rows  : train={len(Xr_train):,}, test={len(Xr_test):,}")
print(f"🧪 Grouped Split Rows : train={len(Xg_train):,}, test={len(Xg_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN ROBUST RANDOM FOREST (BOTH SPLITS)
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    # Same architecture for fair split-to-split comparison
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

print(f"\n⏳ Training Random-Split Model ...")
t0 = time.time()
clf_random = build_model()
clf_random.fit(Xr_train, yr_train)
elapsed_random = time.time() - t0
print(f"✅ Random-Split model complete in {elapsed_random:.1f}s")

print(f"\n⏳ Training Grouped-Split Model ...")
t0 = time.time()
clf_group = build_model()
clf_group.fit(Xg_train, yg_train)
elapsed_group = time.time() - t0
print(f"✅ Grouped-Split model complete in {elapsed_group:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION (SIDE BY SIDE)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(name, model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    acc = accuracy_score(y_eval, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_eval, y_pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_eval, y_proba)
    pr_auc = average_precision_score(y_eval, y_proba)
    fpr, tpr, _ = roc_curve(y_eval, y_proba)
    pr_prec, pr_rec, _ = precision_recall_curve(y_eval, y_proba)

    print(f"\n📋 {name} METRICS:")
    print(f"   Accuracy : {acc * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall   : {recall * 100:.2f}%")
    print(f"   F1-Score : {f1 * 100:.2f}%")
    print(f"   ROC-AUC  : {roc_auc:.4f}")
    print(f"   PR-AUC   : {pr_auc:.4f}")
    print("-" * 30)
    print(classification_report(y_eval, y_pred, target_names=["Wait (0)", "Book Now (1)"]))

    return {
        "name": name,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "pr_prec": pr_prec,
        "pr_rec": pr_rec,
        "y_true": np.asarray(y_eval),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cm": confusion_matrix(y_eval, y_pred),
    }

random_result = evaluate_model("RANDOM SPLIT", clf_random, Xr_test, yr_test)
group_result = evaluate_model("GROUPED SPLIT", clf_group, Xg_test, yg_test)

print("\n📊 QUICK COMPARISON")
print("-" * 62)
print(f"{'Metric':<18}{'Random Split':>18}{'Grouped Split':>18}")
print("-" * 62)
print(f"{'Accuracy':<18}{random_result['acc'] * 100:>17.2f}%{group_result['acc'] * 100:>17.2f}%")
print(f"{'Precision':<18}{random_result['precision'] * 100:>17.2f}%{group_result['precision'] * 100:>17.2f}%")
print(f"{'Recall':<18}{random_result['recall'] * 100:>17.2f}%{group_result['recall'] * 100:>17.2f}%")
print(f"{'F1-Score':<18}{random_result['f1'] * 100:>17.2f}%{group_result['f1'] * 100:>17.2f}%")
print(f"{'ROC-AUC':<18}{random_result['roc_auc']:>18.4f}{group_result['roc_auc']:>18.4f}")
print(f"{'PR-AUC':<18}{random_result['pr_auc']:>18.4f}{group_result['pr_auc']:>18.4f}")
print("-" * 62)

# Save grouped model for realistic deployment behavior
joblib.dump(clf_group, MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERATE VISUALS
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(DARK)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT)
    for s in ax.spines.values(): s.set_edgecolor("#2a2d3a")
    ax.set_title(title, color=CYAN, fontweight="bold", pad=10)

# A: Confusion Matrix
ax_cm = fig.add_subplot(gs[0, 0])
style_ax(ax_cm, "A · Grouped Split Confusion Matrix")
cm = group_result["cm"]
ax_cm.imshow(cm, cmap="Purples", alpha=0.8)
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white", fontweight="bold")

# B: Feature Importance (Real predictors)
ax_fi = fig.add_subplot(gs[0, 1])
style_ax(ax_fi, "B · Feature Importance (No Leakage)")
fi_df = pd.DataFrame({"f": FEATURES, "i": clf_group.feature_importances_}).sort_values("i")
ax_fi.barh(fi_df["f"], fi_df["i"], color=GREEN)

# C: Confidence Distribution
ax_conf = fig.add_subplot(gs[1, 0])
style_ax(ax_conf, "C · Grouped Model Confidence")
g_mask_wait = group_result["y_true"] == 0
g_mask_book = group_result["y_true"] == 1
ax_conf.hist(group_result["y_proba"][g_mask_wait], bins=40, alpha=0.5, color="#ff4b6e", label="Wait")
ax_conf.hist(group_result["y_proba"][g_mask_book], bins=40, alpha=0.5, color=GREEN, label="Book Now")
ax_conf.legend()

# D: Summary Stats
ax_stat = fig.add_subplot(gs[1, 1])
style_ax(ax_stat, "D · Split Comparison Summary")
stats_text = (
    f"Model: RandomForest\n"
    f"Max Depth: 20\n"
    f"Features: {len(FEATURES)}\n"
    f"Leakage: REMOVED\n\n"
    f"Random Accuracy: {random_result['acc'] * 100:.2f}%\n"
    f"Grouped Accuracy: {group_result['acc'] * 100:.2f}%"
)
ax_stat.text(0.5, 0.5, stats_text, color=GOLD, fontsize=14, ha='center', va='center', fontweight='bold')
ax_stat.set_xticks([]); ax_stat.set_yticks([])

plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=DARK)

# Additional chart: metric-by-metric comparison across split strategies
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor(DARK)
ax2.set_facecolor(CARD)
for s in ax2.spines.values():
    s.set_edgecolor("#2a2d3a")

metric_names = ["Accuracy", "Precision", "Recall", "F1"]
random_vals = [
    random_result["acc"] * 100,
    random_result["precision"] * 100,
    random_result["recall"] * 100,
    random_result["f1"] * 100,
]
group_vals = [
    group_result["acc"] * 100,
    group_result["precision"] * 100,
    group_result["recall"] * 100,
    group_result["f1"] * 100,
]

x = np.arange(len(metric_names))
w = 0.35
bars1 = ax2.bar(x - w / 2, random_vals, w, label="Random Split", color="#5c7cfa")
bars2 = ax2.bar(x + w / 2, group_vals, w, label="Grouped Split", color="#00ff9d")

ax2.set_title("Classifier Metrics: Random vs Grouped Split", color=CYAN, fontweight="bold", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(metric_names, color=TEXT)
ax2.set_ylim(0, 100)
ax2.tick_params(colors=TEXT)
ax2.legend(facecolor=CARD, edgecolor="#2a2d3a", labelcolor=TEXT)

for bars in (bars1, bars2):
    for b in bars:
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.6,
            f"{b.get_height():.2f}%",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig(PLOT_COMPARE_PATH, dpi=150, bbox_inches="tight", facecolor=DARK)

# Additional chart: ROC and Precision-Recall curves for both split strategies
fig3, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
fig3.patch.set_facecolor(DARK)

for ax in (ax_roc, ax_pr):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT)
    for s in ax.spines.values():
        s.set_edgecolor("#2a2d3a")

ax_roc.plot(
    random_result["fpr"],
    random_result["tpr"],
    color="#5c7cfa",
    linewidth=2,
    label=f"Random (AUC={random_result['roc_auc']:.4f})",
)
ax_roc.plot(
    group_result["fpr"],
    group_result["tpr"],
    color="#00ff9d",
    linewidth=2,
    label=f"Grouped (AUC={group_result['roc_auc']:.4f})",
)
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#7a7f93", linewidth=1.2, label="Baseline")
ax_roc.set_title("ROC Curve Comparison", color=CYAN, fontweight="bold", pad=10)
ax_roc.set_xlabel("False Positive Rate", color=TEXT)
ax_roc.set_ylabel("True Positive Rate", color=TEXT)
ax_roc.set_xlim(0, 1)
ax_roc.set_ylim(0, 1)
ax_roc.legend(facecolor=CARD, edgecolor="#2a2d3a", labelcolor=TEXT)

ax_pr.plot(
    random_result["pr_rec"],
    random_result["pr_prec"],
    color="#5c7cfa",
    linewidth=2,
    label=f"Random (AP={random_result['pr_auc']:.4f})",
)
ax_pr.plot(
    group_result["pr_rec"],
    group_result["pr_prec"],
    color="#00ff9d",
    linewidth=2,
    label=f"Grouped (AP={group_result['pr_auc']:.4f})",
)
baseline = float(np.mean(group_result["y_true"]))
ax_pr.hlines(
    y=baseline,
    xmin=0,
    xmax=1,
    colors="#7a7f93",
    linestyles="--",
    linewidth=1.2,
    label=f"Baseline={baseline:.3f}",
)
ax_pr.set_title("Precision-Recall Curve Comparison", color=CYAN, fontweight="bold", pad=10)
ax_pr.set_xlabel("Recall", color=TEXT)
ax_pr.set_ylabel("Precision", color=TEXT)
ax_pr.set_xlim(0, 1)
ax_pr.set_ylim(0, 1)
ax_pr.legend(facecolor=CARD, edgecolor="#2a2d3a", labelcolor=TEXT)

plt.tight_layout()
plt.savefig(PLOT_CURVES_PATH, dpi=150, bbox_inches="tight", facecolor=DARK)

print(f"\n✅ Model saved to: {MODEL_PATH}")
print(f"✅ Report saved to: {PLOT_PATH}")
print(f"✅ Split comparison chart saved to: {PLOT_COMPARE_PATH}")
print(f"✅ ROC/PR comparison chart saved to: {PLOT_CURVES_PATH}")
print("=" * 62)
