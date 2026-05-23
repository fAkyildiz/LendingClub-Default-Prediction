"""Generate a presentation-ready PDF from the LendingClub project results."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUTPUT = r"C:\Users\Administrator\Desktop\DataMiningProject\Presentation_Script.pdf"

# ── Colour palette ──────────────────────────────────────────────────────────
BLUE   = "#1E3A5F"   # dark navy  – headings / bullets
ACCENT = "#2E86AB"   # mid blue   – section banners
LIGHT  = "#D6EAF8"   # pale blue  – background boxes
RED    = "#E84C4C"
GREEN  = "#2CA02C"
GREY   = "#F4F6F9"
WHITE  = "#FFFFFF"


# ── Helper utilities ─────────────────────────────────────────────────────────

def new_slide(bg=WHITE):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))   # 16:9 @ 100 dpi ~ 1333×750
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    return fig, ax


def banner(ax, text, y=0.92, color=ACCENT):
    ax.add_patch(FancyBboxPatch((0, y - 0.04), 1, 0.12,
                                boxstyle="round,pad=0", linewidth=0,
                                facecolor=color, zorder=0))
    ax.text(0.5, y + 0.02, text, ha="center", va="center",
            fontsize=20, fontweight="bold", color=WHITE, zorder=1)


def section_badge(ax, label, x=0.05, y=0.91):
    ax.add_patch(FancyBboxPatch((x, y - 0.025), 0.16, 0.05,
                                boxstyle="round,pad=0.01", linewidth=0,
                                facecolor=ACCENT, zorder=0))
    ax.text(x + 0.08, y, label, ha="center", va="center",
            fontsize=9, color=WHITE, fontweight="bold", zorder=1)


def bullets(ax, items, y_start=0.75, x=0.08, step=0.09, fontsize=13,
            color=BLUE, indent=0):
    for i, line in enumerate(items):
        dot_x = x + indent
        ax.plot(dot_x, y_start - i * step, "o", color=ACCENT,
                markersize=5, zorder=2)
        ax.text(dot_x + 0.025, y_start - i * step, line,
                va="center", fontsize=fontsize, color=color)


def sub_bullets(ax, items, y_start, x=0.12, step=0.065, fontsize=11):
    for i, line in enumerate(items):
        ax.plot(x, y_start - i * step, "s", color=RED,
                markersize=4, zorder=2)
        ax.text(x + 0.025, y_start - i * step, line,
                va="center", fontsize=fontsize, color="#444")


def divider(ax, y, color=LIGHT):
    ax.axhline(y, color=color, linewidth=2, xmin=0.05, xmax=0.95)


def metric_box(ax, x, y, w, h, label, value, sub="", bg=LIGHT):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.01", linewidth=1,
                                edgecolor=ACCENT, facecolor=bg))
    ax.text(x + w / 2, y + h * 0.72, value, ha="center", va="center",
            fontsize=19, fontweight="bold", color=BLUE)
    ax.text(x + w / 2, y + h * 0.35, label, ha="center", va="center",
            fontsize=10, color="#555")
    if sub:
        ax.text(x + w / 2, y + h * 0.12, sub, ha="center", va="center",
                fontsize=8, color="#888")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 1 – Title
# ─────────────────────────────────────────────────────────────────────────────
def slide_title():
    fig, ax = new_slide(bg=BLUE)
    ax.add_patch(patches.Rectangle((0, 0.55), 1, 0.45,
                                   color="#16304F", zorder=0))
    ax.text(0.5, 0.82, "Predicting Loan Default Risk",
            ha="center", va="center", fontsize=30, fontweight="bold",
            color=WHITE)
    ax.text(0.5, 0.68, "Using LendingClub Data",
            ha="center", va="center", fontsize=22, color="#A8D1F5")

    ax.add_patch(patches.Rectangle((0.2, 0.50), 0.6, 0.003,
                                   color=ACCENT, zorder=1))

    info = [
        ("Course",   "Data Mining"),
        ("Type",     "Binary Classification"),
        ("Dataset",  "LendingClub  (2007 – 2018 Q4)"),
        ("Records",  "1,345,310 terminal loans"),
        ("Features", "132 predictors (after preprocessing)"),
    ]
    for i, (lbl, val) in enumerate(info):
        y = 0.43 - i * 0.075
        ax.text(0.30, y, lbl + ":", ha="right", va="center",
                fontsize=12, color="#A8D1F5", fontweight="bold")
        ax.text(0.33, y, val, ha="left", va="center",
                fontsize=12, color=WHITE)

    ax.text(0.5, 0.04, "Ferit Akyildiz  |  2026",
            ha="center", fontsize=10, color="#888", style="italic")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 2 – Problem Statement & Research Objective
# ─────────────────────────────────────────────────────────────────────────────
def slide_problem():
    fig, ax = new_slide()
    banner(ax, "1 · Introduction & Problem Statement")

    # Problem box
    ax.add_patch(FancyBboxPatch((0.04, 0.56), 0.43, 0.28,
                                boxstyle="round,pad=0.01", linewidth=1,
                                edgecolor=ACCENT, facecolor=LIGHT))
    ax.text(0.255, 0.88, "The Problem", ha="center", fontsize=13,
            fontweight="bold", color=BLUE)
    ax.text(0.255, 0.80, "LendingClub assigns grades A–G to each loan,\n"
            "but grades alone don't capture true default\n"
            "probability at the individual loan level.",
            ha="center", va="center", fontsize=11, color="#333",
            wrap=True)
    ax.text(0.255, 0.62,
            "Can a machine learning model predict whether\n"
            "a loan will be Fully Paid or Charged Off —\n"
            "and beat LendingClub's own grading system?",
            ha="center", va="center", fontsize=11.5,
            color=BLUE, fontweight="bold")

    # Objective box
    ax.add_patch(FancyBboxPatch((0.52, 0.56), 0.44, 0.28,
                                boxstyle="round,pad=0.01", linewidth=1,
                                edgecolor=ACCENT, facecolor=LIGHT))
    ax.text(0.74, 0.88, "Research Objectives", ha="center", fontsize=13,
            fontweight="bold", color=BLUE)
    obj = [
        "Build a binary classifier (Fully Paid vs. Charged Off)",
        "Use only features available at loan origination",
        "Evaluate model lift vs. LendingClub's grade signal",
        "Identify key default drivers via SHAP explainability",
    ]
    for i, o in enumerate(obj):
        ax.plot(0.55, 0.81 - i * 0.065, "o", color=ACCENT, markersize=5)
        ax.text(0.57, 0.81 - i * 0.065, o, va="center", fontsize=10.5,
                color="#333")

    # Why it matters
    ax.add_patch(FancyBboxPatch((0.04, 0.18), 0.92, 0.32,
                                boxstyle="round,pad=0.01", linewidth=0,
                                facecolor="#EBF5FB"))
    ax.text(0.5, 0.49, "Why It Matters", ha="center", fontsize=13,
            fontweight="bold", color=BLUE)
    why = [
        "Consumer lending involves substantial financial risk — misclassifying defaults is costly",
        "A better risk signal allows lenders to minimize losses while extending credit fairly",
        "Interpretable models build trust with regulators and borrowers alike",
    ]
    for i, w in enumerate(why):
        ax.plot(0.07, 0.43 - i * 0.07, "o", color=GREEN, markersize=5)
        ax.text(0.09, 0.43 - i * 0.07, w, va="center", fontsize=11, color="#333")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 3 – Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────
def slide_dataset():
    fig, ax = new_slide()
    banner(ax, "2 · Dataset Overview")

    # Four stat boxes at top
    boxes = [
        ("Raw file size", "1.68 GB", "1 CSV, 151 columns"),
        ("Total records", "2,260,701", "Accepted loans 2007–2018"),
        ("After filtering", "1,345,310", "Fully Paid + Charged Off only"),
        ("Default rate", "~20%", "Class imbalance present"),
    ]
    for i, (lbl, val, sub) in enumerate(boxes):
        metric_box(ax, 0.04 + i * 0.24, 0.60, 0.21, 0.22, lbl, val, sub)

    divider(ax, 0.57)

    # Feature groups
    ax.text(0.08, 0.52, "Key Feature Groups (at origination):", fontsize=12,
            fontweight="bold", color=BLUE)
    groups = [
        ("Credit Profile",      "FICO, inquiries (last 6 mo.), delinquencies, revolving utilisation, pub records"),
        ("Loan Characteristics","Loan amount, interest rate, term, grade, sub-grade, purpose"),
        ("Borrower Profile",    "Annual income, DTI, employment length, home ownership, state"),
        ("Engineered",          "Loan-to-income ratio, credit age (months), FICO midpoint average"),
    ]
    for i, (grp, desc) in enumerate(groups):
        y = 0.44 - i * 0.08
        ax.text(0.09, y, grp + ":", fontsize=11, fontweight="bold",
                color=ACCENT, va="center")
        ax.text(0.30, y, desc, fontsize=10.5, color="#333", va="center")

    divider(ax, 0.10)
    ax.text(0.5, 0.06,
            "Post-origination columns (recoveries, total_pymnt, out_prncp, etc.) excluded to prevent data leakage",
            ha="center", fontsize=10, color=RED, style="italic")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 4 – EDA: Class Imbalance & Grade Analysis
# ─────────────────────────────────────────────────────────────────────────────
def slide_eda_class():
    fig, ax = new_slide()
    banner(ax, "3 · Exploratory Data Analysis — Class & Grade")

    # Class bar chart (manual)
    ax2 = fig.add_axes([0.06, 0.22, 0.36, 0.52])
    vals = [1076751, 268559]
    colors = ["#4C9BE8", "#E84C4C"]
    labels = ["Fully Paid\n(80.0%)", "Charged Off\n(20.0%)"]
    bars = ax2.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
    ax2.set_title("Class Distribution (n = 1,345,310)", fontsize=11)
    ax2.set_ylabel("Count")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 10000,
                 f"{v:,}", ha="center", fontsize=10)
    ax2.set_facecolor(GREY)
    ax2.grid(axis="y", alpha=0.4)

    # Grade default rates (from EDA results)
    ax3 = fig.add_axes([0.52, 0.22, 0.44, 0.52])
    grades = ["A", "B", "C", "D", "E", "F", "G"]
    rates  = [0.054, 0.121, 0.191, 0.277, 0.363, 0.431, 0.472]
    ax3.bar(grades, rates, color="#4C9BE8", edgecolor="white")
    ax3.set_title("Default Rate by LendingClub Grade", fontsize=11)
    ax3.set_ylabel("Default Rate")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    for i, r in enumerate(rates):
        ax3.text(i, r + 0.008, f"{r:.1%}", ha="center", fontsize=9)
    ax3.set_facecolor(GREY)
    ax3.grid(axis="y", alpha=0.4)

    # Key takeaways
    ax.add_patch(FancyBboxPatch((0.04, 0.05), 0.92, 0.13,
                                boxstyle="round,pad=0.01", linewidth=0,
                                facecolor=LIGHT))
    ax.text(0.5, 0.17, "Key Takeaways", ha="center", fontsize=11,
            fontweight="bold", color=BLUE)
    takes = [
        "20% default rate requires imbalance handling (SMOTE / class weights)",
        "Grade monotonically separates risk: A (5.4%) → G (47.2%) — a strong ordinal signal",
        "Grade-only ROC-AUC = 0.680 — our model target is to exceed this benchmark",
    ]
    for i, t in enumerate(takes):
        ax.plot(0.06 + i * 0.33, 0.08, "o", color=ACCENT, markersize=5)
        ax.text(0.08 + i * 0.33, 0.08, t, va="center", fontsize=9.5,
                color="#333", wrap=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 5 – EDA: Feature Insights
# ─────────────────────────────────────────────────────────────────────────────
def slide_eda_features():
    fig, ax = new_slide()
    banner(ax, "3 · Exploratory Data Analysis — Feature Insights")

    # Correlation table
    ax.text(0.05, 0.83, "Top Correlations with Default (target):",
            fontsize=13, fontweight="bold", color=BLUE)
    corr_data = [
        ("int_rate",        "+0.31", "Higher interest = higher risk"),
        ("sub_grade",       "+0.30", "Grade encoding captures risk ordering"),
        ("grade",           "+0.28", "Coarse version of sub_grade"),
        ("dti",             "+0.11", "Higher leverage → more defaults"),
        ("fico_range_low",  "-0.13", "Lower FICO → more defaults"),
        ("annual_inc (log)","-0.09", "Higher income → fewer defaults"),
    ]
    for i, (feat, corr, note) in enumerate(corr_data):
        y = 0.74 - i * 0.07
        color = RED if corr.startswith("+") else GREEN
        ax.add_patch(FancyBboxPatch((0.05, y - 0.025), 0.90, 0.05,
                                    boxstyle="round,pad=0", linewidth=0,
                                    facecolor=GREY if i % 2 == 0 else WHITE))
        ax.text(0.06, y, feat, va="center", fontsize=11,
                fontfamily="monospace", color=BLUE)
        ax.text(0.30, y, corr, va="center", fontsize=12,
                fontweight="bold", color=color)
        ax.text(0.40, y, note, va="center", fontsize=10.5, color="#444")

    divider(ax, 0.28)

    # Other EDA bullets
    ax.text(0.05, 0.24, "Other Notable Findings:", fontsize=12,
            fontweight="bold", color=BLUE)
    others = [
        "60-month loans default at ~26% vs. ~16% for 36-month loans",
        "Small-business and moving loans show highest default rates by purpose",
        "Rejected applicants have higher DTI & lower FICO vs. accepted borrowers",
        "73 of 151 columns had missing values; 35 columns dropped (>50% missing)",
    ]
    for i, o in enumerate(others):
        ax.plot(0.07, 0.17 - i * 0.06, "o", color=ACCENT, markersize=5)
        ax.text(0.09, 0.17 - i * 0.06, o, va="center", fontsize=11, color="#333")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 6 – Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def slide_preprocessing():
    fig, ax = new_slide()
    banner(ax, "4 · Preprocessing Pipeline")

    steps = [
        ("1", "Column Selection",
         "Exclude 37 post-origination + 10 non-predictive columns at load time\n"
         "151 raw columns → 102 loaded"),
        ("2", "Target Filtering",
         "Keep only Fully Paid / Charged Off rows\n"
         "2,260,701 → 1,345,310 rows  |  default rate: 19.96%"),
        ("3", "Missing Value Drop",
         "Drop 35 columns with >50% missing rate\n"
         "(e.g. annual_inc_joint, dti_joint, mths_since_last_major_derog)"),
        ("4", "Feature Engineering",
         "loan_to_income = loan_amnt / annual_inc\n"
         "credit_age_months = issue_d - earliest_cr_line\n"
         "fico_avg = (fico_range_low + fico_range_high) / 2"),
        ("5", "Encoding",
         "Ordinal: grade (1–7), sub_grade (1–35), emp_length (0–10), term (36/60)\n"
         "One-hot: home_ownership, purpose, verification_status, addr_state"),
        ("6", "Imputation",
         "Median imputation for 42 remaining numeric columns with NaN\n"
         "→ 0 nulls remaining"),
    ]

    cols = 2
    for i, (num, title, body) in enumerate(steps):
        col = i % cols
        row = i // cols
        x = 0.04 + col * 0.50
        y = 0.72 - row * 0.24

        ax.add_patch(FancyBboxPatch((x, y - 0.16), 0.45, 0.20,
                                    boxstyle="round,pad=0.01", linewidth=1,
                                    edgecolor=ACCENT, facecolor=GREY))
        ax.add_patch(FancyBboxPatch((x + 0.005, y + 0.01), 0.04, 0.04,
                                    boxstyle="round,pad=0.005", linewidth=0,
                                    facecolor=ACCENT))
        ax.text(x + 0.025, y + 0.025, num, ha="center", va="center",
                fontsize=12, fontweight="bold", color=WHITE)
        ax.text(x + 0.055, y + 0.025, title, va="center",
                fontsize=11, fontweight="bold", color=BLUE)
        ax.text(x + 0.02, y - 0.07, body, va="center",
                fontsize=9.5, color="#444", linespacing=1.5)

    # Final shape
    ax.add_patch(FancyBboxPatch((0.15, 0.03), 0.70, 0.065,
                                boxstyle="round,pad=0.01", linewidth=0,
                                facecolor=ACCENT))
    ax.text(0.5, 0.063, "Final dataset:  1,345,310 rows  ×  133 columns  |  0 nulls  |  all numeric",
            ha="center", va="center", fontsize=12, color=WHITE, fontweight="bold")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 7 – Modeling Approach
# ─────────────────────────────────────────────────────────────────────────────
def slide_modeling():
    fig, ax = new_slide()
    banner(ax, "5 · Modeling Approach")

    # Train/test split
    ax.text(0.05, 0.82, "Train / Test Split", fontsize=13, fontweight="bold", color=BLUE)
    ax.text(0.05, 0.76,
            "Stratified 80/20 split  →  Train: 1,076,248 rows  |  Test: 269,062 rows\n"
            "Both sets maintain 19.96% default rate",
            fontsize=11, color="#333")

    divider(ax, 0.71)

    # Three models
    ax.text(0.05, 0.68, "Three Models Trained:", fontsize=13,
            fontweight="bold", color=BLUE)

    models = [
        ("Logistic Regression\n(Baseline)",
         "class_weight='balanced'\nSAGA solver, C=0.1\nStandardScaler pipeline",
         "Simple, interpretable baseline\nHandles imbalance via weights"),
        ("Random Forest",
         "300 trees, max_depth=12\nmin_samples_leaf=50\nclass_weight='balanced_subsample'",
         "Non-linear, robust to outliers\nImbalance via per-tree resampling"),
        ("XGBoost\n(Best Model)",
         "500 estimators, lr=0.05\nmax_depth=6, subsample=0.8\nscale_pos_weight=4.01",
         "Gradient boosting with\nnative imbalance correction"),
    ]

    for i, (title, params, rationale) in enumerate(models):
        x = 0.04 + i * 0.325
        is_best = i == 2
        color = ACCENT if is_best else "#AAA"
        ax.add_patch(FancyBboxPatch((x, 0.18), 0.305, 0.46,
                                    boxstyle="round,pad=0.01", linewidth=2,
                                    edgecolor=color, facecolor=GREY))
        ax.text(x + 0.152, 0.63, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=BLUE)
        divider(ax, 0.595)
        ax.text(x + 0.01, 0.565, "Parameters:", fontsize=9.5,
                fontweight="bold", color="#555")
        ax.text(x + 0.01, 0.51, params, fontsize=9, color="#333",
                linespacing=1.5)
        ax.text(x + 0.01, 0.40, "Why:", fontsize=9.5,
                fontweight="bold", color="#555")
        ax.text(x + 0.01, 0.345, rationale, fontsize=9, color="#333",
                linespacing=1.5)
        if is_best:
            ax.add_patch(FancyBboxPatch((x + 0.04, 0.20), 0.22, 0.04,
                                        boxstyle="round,pad=0.005",
                                        linewidth=0, facecolor=RED))
            ax.text(x + 0.152, 0.22, "Best Performer", ha="center",
                    va="center", fontsize=9, color=WHITE, fontweight="bold")

    # SMOTE note
    ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.10,
                                boxstyle="round,pad=0.01", linewidth=0,
                                facecolor=LIGHT))
    ax.text(0.5, 0.115, "Imbalance Strategy Comparison", ha="center",
            fontsize=10, fontweight="bold", color=BLUE)
    ax.text(0.5, 0.065,
            "SMOTE (synthetic over-sampling) was tested against scale_pos_weight.\n"
            "SMOTE ROC-AUC = 0.7314 vs. XGBoost native = 0.7342 — native weighting preferred.",
            ha="center", fontsize=10, color="#333")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 8 – Results
# ─────────────────────────────────────────────────────────────────────────────
def slide_results():
    fig, ax = new_slide()
    banner(ax, "6 · Model Results")

    # Metrics table header
    headers = ["Model", "ROC-AUC", "PR-AUC", "F1 (t=0.5)"]
    rows = [
        ("Logistic Regression (balanced)", "0.7186", "0.3828", "0.4359"),
        ("Random Forest",                  "0.7175", "0.3859", "0.4326"),
        ("XGBoost",                        "0.7342", "0.4095", "0.4461"),
        ("XGBoost (optimal threshold)",    "0.7342", "0.4095", "0.4485"),
        ("Grade-only baseline",            "0.6801", "—",      "—"),
    ]

    col_x = [0.05, 0.48, 0.61, 0.74]
    col_w = 0.42

    # Header row
    ax.add_patch(FancyBboxPatch((0.04, 0.73), 0.92, 0.06,
                                boxstyle="round,pad=0", linewidth=0,
                                facecolor=ACCENT))
    for j, h in enumerate(headers):
        ax.text(col_x[j] + (0.12 if j == 0 else 0.04), 0.76, h,
                ha="left" if j == 0 else "center",
                fontsize=11, fontweight="bold", color=WHITE)

    # Data rows
    highlight_rows = {2, 3}   # XGBoost rows
    for i, (model, *vals) in enumerate(rows):
        y = 0.67 - i * 0.083
        bg = LIGHT if i in highlight_rows else (GREY if i % 2 == 0 else WHITE)
        ax.add_patch(FancyBboxPatch((0.04, y - 0.025), 0.92, 0.055,
                                    boxstyle="round,pad=0", linewidth=0,
                                    facecolor=bg))
        ax.text(col_x[0], y + 0.005, model, va="center",
                fontsize=10.5, color=BLUE,
                fontweight="bold" if i in highlight_rows else "normal")
        for j, v in enumerate(vals):
            col = RED if (i in highlight_rows and j == 0) else "#333"
            ax.text(col_x[j + 1] + 0.04, y + 0.005, v,
                    ha="center", va="center", fontsize=11,
                    color=col,
                    fontweight="bold" if i in highlight_rows else "normal")

    divider(ax, 0.24)

    # CV note
    ax.text(0.5, 0.20, "5-Fold Cross-Validation (ROC-AUC):",
            ha="center", fontsize=11, fontweight="bold", color=BLUE)
    cv_data = [
        ("Logistic Regression", "0.7178 ± 0.0014"),
        ("Random Forest",       "0.7172 ± 0.0013"),
        ("XGBoost",             "0.7333 ± 0.0009"),
    ]
    for i, (m, cv) in enumerate(cv_data):
        x = 0.08 + i * 0.31
        ax.add_patch(FancyBboxPatch((x, 0.06), 0.27, 0.10,
                                    boxstyle="round,pad=0.01", linewidth=1,
                                    edgecolor=ACCENT, facecolor=GREY))
        ax.text(x + 0.135, 0.125, m, ha="center", fontsize=9.5, color="#555")
        col = RED if i == 2 else BLUE
        ax.text(x + 0.135, 0.085, cv, ha="center", fontsize=11,
                fontweight="bold", color=col)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 9 – SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────
def slide_shap():
    fig, ax = new_slide()
    banner(ax, "7 · SHAP Explainability — What Drives Defaults?")

    # Horizontal bar chart of top SHAP features
    ax2 = fig.add_axes([0.04, 0.12, 0.42, 0.68])
    features = [
        "sub_grade", "term", "grade", "dti", "loan_to_income",
        "acc_open_past_24mths", "int_rate", "home_ownership_RENT",
        "fico_range_low", "mort_acc",
    ]
    shap_vals = [0.295, 0.185, 0.174, 0.129, 0.109,
                 0.101, 0.081, 0.055, 0.055, 0.052]
    colors_bar = [ACCENT] * 3 + [RED] * 2 + [ACCENT] * 3 + [GREEN] * 2
    bars = ax2.barh(features[::-1], shap_vals[::-1],
                    color=colors_bar[::-1], edgecolor="white")
    ax2.set_title("Top 10 Features — Mean |SHAP| (XGBoost)", fontsize=11)
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_facecolor(GREY)
    ax2.grid(axis="x", alpha=0.4)
    for bar, v in zip(bars, shap_vals[::-1]):
        ax2.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=9)

    # Interpretation bullets
    ax.text(0.52, 0.82, "Feature Interpretations", fontsize=13,
            fontweight="bold", color=BLUE)
    interp = [
        ("sub_grade / grade / int_rate",
         "LendingClub's own risk signal is the strongest predictor\n"
         "— but the model adds information beyond it"),
        ("term",
         "60-month loans carry significantly higher default risk\n"
         "— second most influential feature"),
        ("dti + loan_to_income",
         "Borrower leverage at origination is a key risk driver;\n"
         "our engineered ratio adds signal beyond raw DTI"),
        ("acc_open_past_24mths",
         "Recent credit-seeking behaviour signals financial stress"),
        ("fico_range_low",
         "Lower FICO → higher default; negative SHAP direction\n"
         "(as expected — strong inverse relationship)"),
        ("mort_acc",
         "More mortgages → lower default risk (wealthier borrowers)"),
    ]
    for i, (feat, desc) in enumerate(interp):
        y = 0.74 - i * 0.105
        ax.add_patch(FancyBboxPatch((0.51, y - 0.045), 0.46, 0.07,
                                    boxstyle="round,pad=0.01", linewidth=0,
                                    facecolor=GREY if i % 2 == 0 else WHITE))
        ax.text(0.53, y, feat, va="center", fontsize=10,
                fontweight="bold", color=ACCENT)
        ax.text(0.53, y - 0.025, desc, va="center", fontsize=9.5, color="#333")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 10 – Model vs. Grade Benchmark
# ─────────────────────────────────────────────────────────────────────────────
def slide_vs_grade():
    fig, ax = new_slide()
    banner(ax, "8 · Model vs. LendingClub Grade")

    # Three big comparison metrics
    metric_box(ax, 0.05, 0.50, 0.27, 0.28,
               "Grade-only ROC-AUC", "0.680", "treating grade 1–7 as score", bg="#FDEDEC")
    metric_box(ax, 0.365, 0.50, 0.27, 0.28,
               "XGBoost (no grade) AUC", "0.733",
               "other features alone vs grade", bg=LIGHT)
    metric_box(ax, 0.68, 0.50, 0.27, 0.28,
               "XGBoost Full AUC", "0.734",
               "lift over grade: +0.054", bg=LIGHT)

    # Arrow showing lift
    ax.annotate("", xy=(0.34, 0.645), xytext=(0.23, 0.645),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.text(0.285, 0.672, "+0.053", ha="center", fontsize=11,
            fontweight="bold", color=GREEN)

    ax.annotate("", xy=(0.655, 0.645), xytext=(0.545, 0.645),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.text(0.60, 0.672, "+0.001", ha="center", fontsize=11,
            fontweight="bold", color="#AAA")

    divider(ax, 0.47)

    ax.text(0.5, 0.43, "What This Tells Us", ha="center", fontsize=13,
            fontweight="bold", color=BLUE)
    insights = [
        "The model provides a meaningful +5.4% AUC lift over LendingClub's grades alone",
        "Even without grade/sub_grade, other borrower features achieve 73.3% AUC — "
        "beating grade-only by +5.3%",
        "Adding grade back improves AUC only marginally (+0.1%) — most grade signal "
        "is already captured by correlated features",
        "Conclusion: a data-driven model significantly outperforms the existing grade "
        "system as a default risk signal",
    ]
    for i, ins in enumerate(insights):
        ax.plot(0.07, 0.36 - i * 0.075, "o",
                color=GREEN if i < 3 else RED, markersize=6)
        ax.text(0.09, 0.36 - i * 0.075, ins,
                va="center", fontsize=10.5, color="#333")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 11 – Future Work
# ─────────────────────────────────────────────────────────────────────────────
def slide_future():
    fig, ax = new_slide()
    banner(ax, "9 · Future Work & Limitations")

    ax.text(0.05, 0.82, "Potential Next Steps", fontsize=13,
            fontweight="bold", color=BLUE)
    future = [
        ("Hyperparameter Tuning",
         "Bayesian search over XGBoost parameters (max_depth, min_child_weight,\n"
         "gamma, learning_rate) — expected additional AUC improvement of 0.01–0.02"),
        ("Neural Networks / TabNet",
         "Deep tabular models may capture higher-order feature interactions\n"
         "that tree models miss in the 132-feature space"),
        ("Calibration",
         "Platt scaling or isotonic regression to produce well-calibrated probabilities\n"
         "— currently AUC is strong but probability values need calibration check"),
        ("Business Threshold Optimisation",
         "Tie decision threshold to actual financial cost (loss given default, LGD)\n"
         "rather than maximising F1 alone — different cost for FP vs. FN"),
        ("Temporal Validation",
         "Validate on a strict time-based split (train 2007–2015, test 2016–2018)\n"
         "— prevents any future data leaking through time"),
        ("NLP on Loan Purpose Text",
         "The 'desc' / 'title' free-text columns were dropped — NLP embeddings\n"
         "could extract additional signal"),
    ]
    for i, (title, body) in enumerate(future):
        col = i % 2
        row = i // 2
        x = 0.04 + col * 0.495
        y = 0.72 - row * 0.19
        ax.add_patch(FancyBboxPatch((x, y - 0.13), 0.465, 0.15,
                                    boxstyle="round,pad=0.01", linewidth=1,
                                    edgecolor=ACCENT, facecolor=GREY))
        ax.text(x + 0.01, y + 0.005, title, va="center",
                fontsize=10.5, fontweight="bold", color=BLUE)
        ax.text(x + 0.01, y - 0.065, body, va="center",
                fontsize=9.5, color="#333", linespacing=1.5)

    # Limitations
    divider(ax, 0.14)
    ax.text(0.5, 0.11, "Key Limitations", ha="center", fontsize=11,
            fontweight="bold", color=RED)
    lims = [
        "Data covers 2007–2018 — economic conditions may not generalise to current market",
        "Rejected applications have far fewer features — full selection-bias analysis not possible",
        "SMOTE on 1M+ rows is computationally expensive — class weights used as practical alternative",
    ]
    for i, l in enumerate(lims):
        x = 0.06 + i * 0.32
        ax.plot(x, 0.055, "^", color=RED, markersize=6)
        ax.text(x + 0.02, 0.055, l, va="center", fontsize=9, color="#555",
                wrap=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Slide 12 – Conclusion
# ─────────────────────────────────────────────────────────────────────────────
def slide_conclusion():
    fig, ax = new_slide(bg=BLUE)

    ax.text(0.5, 0.88, "Summary & Conclusions",
            ha="center", fontsize=26, fontweight="bold", color=WHITE)
    ax.add_patch(patches.Rectangle((0.2, 0.845), 0.6, 0.003,
                                   color=ACCENT, zorder=1))

    conclusions = [
        ("Dataset", "1.34M terminal loans, 132 features, 20% default rate"),
        ("EDA",     "Grade, interest rate, DTI, and FICO are the dominant risk signals"),
        ("Pipeline","Post-origination leakage prevented; 3 engineered features added"),
        ("Modeling","XGBoost (ROC-AUC 0.734) outperforms LR and Random Forest"),
        ("vs. Grade","Model lifts AUC by +5.4% over LendingClub's grade-only baseline"),
        ("SHAP",    "sub_grade, term, dti, and loan_to_income are top model drivers"),
    ]
    for i, (lbl, val) in enumerate(conclusions):
        y = 0.74 - i * 0.095
        ax.add_patch(FancyBboxPatch((0.08, y - 0.038), 0.84, 0.075,
                                    boxstyle="round,pad=0.01", linewidth=0,
                                    facecolor="#16304F"))
        ax.add_patch(FancyBboxPatch((0.08, y - 0.038), 0.16, 0.075,
                                    boxstyle="round,pad=0", linewidth=0,
                                    facecolor=ACCENT))
        ax.text(0.16, y + 0.0, lbl, ha="center", va="center",
                fontsize=11, fontweight="bold", color=WHITE)
        ax.text(0.27, y + 0.0, val, va="center",
                fontsize=11, color="#C8E6FF")

    ax.text(0.5, 0.06,
            "A machine learning approach to credit default prediction adds measurable value\n"
            "beyond traditional rule-based grading systems — with full interpretability via SHAP.",
            ha="center", fontsize=12, color="#A8D1F5", style="italic")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Render all slides to PDF
# ─────────────────────────────────────────────────────────────────────────────
slides = [
    slide_title,
    slide_problem,
    slide_dataset,
    slide_eda_class,
    slide_eda_features,
    slide_preprocessing,
    slide_modeling,
    slide_results,
    slide_shap,
    slide_vs_grade,
    slide_future,
    slide_conclusion,
]

with PdfPages(OUTPUT) as pdf:
    for fn in slides:
        fig = fn()
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

print(f"Saved: {OUTPUT}  ({len(slides)} slides)")
