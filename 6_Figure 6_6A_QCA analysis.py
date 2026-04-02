# -*- coding: utf-8 -*-
# Batch fsQCA (binary 0/1) analysis
# pip install pandas sympy openpyxl matplotlib

import pandas as pd
import numpy as np
from pathlib import Path
from sympy import symbols
from sympy.logic.boolalg import SOPform, simplify_logic
import matplotlib.pyplot as plt
import os

# ===== 0) Global parameters =====
N_CUT = 1
INCL_CUT = 0.80

# ===== 1) Input & output paths =====
base_dir = os.path.dirname(__file__) 
data_file = '0_Dataset.csv'           
outdir = os.path.join(base_dir, 'results')
os.makedirs(outdir, exist_ok=True)

# ===== 2) Data loading =====
def load_and_binarize(csv_path):
    """Load CSV and binarize all variables (values > 0 -> 1)"""
    df = pd.read_csv(csv_path, encoding="utf-8")

    outcome_candidates = [c for c in df.columns if c.lower() in ("y", "outcome", "result")]
    if not outcome_candidates:
        raise ValueError("Outcome column not found. Use y / outcome / result.")

    Ycol = outcome_candidates[0]
    conds = [c for c in df.columns if c != Ycol]

    df_bin = df.copy()
    for c in [*conds, Ycol]:
        df_bin[c] = (pd.to_numeric(df_bin[c], errors="coerce").fillna(0) > 0).astype(int)

    return df_bin, Ycol, conds

# ===== 3) Truth table =====
def truth_table_and_keepers(df_bin, Ycol, conds):
    g = df_bin.groupby(conds, dropna=False)
    tt = g[Ycol].agg(n="count", n_suff="sum").reset_index()

    tt["consistency"] = tt["n_suff"] / tt["n"].replace(0, np.nan)
    total_y1 = df_bin[Ycol].sum()
    tt["coverage"] = tt["n_suff"] / (total_y1 if total_y1 else 1)

    keepers = tt.query("n >= @N_CUT and consistency >= @INCL_CUT").copy()
    return tt, keepers

# ===== 4) Minimization =====
def minimize_solution(keepers, conds):
    mins = keepers[conds].values.tolist()
    if not mins:
        return "No solution under current thresholds."

    sym_vars = symbols(" ".join(conds))
    expr = SOPform(sym_vars, mins, [])
    expr_simplified = simplify_logic(expr, form="dnf")
    return str(expr_simplified)

# ===== 5) Necessity analysis =====
def necessity_tables(df_bin, Ycol, conds):
    rows = []
    y = df_bin[Ycol]

    for c in conds:
        x = df_bin[c]
        x_not = 1 - x

        inter_pos = np.minimum(x, y).sum()
        inter_neg = np.minimum(x_not, y).sum()

        rows.append({
            "condition": c,
            "type": "X",
            "consistency": inter_pos / y.sum() if y.sum() else np.nan,
            "coverage": inter_pos / x.sum() if x.sum() else np.nan,
        })

        rows.append({
            "condition": f"~{c}",
            "type": "~X",
            "consistency": inter_neg / y.sum() if y.sum() else np.nan,
            "coverage": inter_neg / x_not.sum() if x_not.sum() else np.nan,
        })

    return pd.DataFrame(rows).round(4)

# ===== 6) Sufficiency singletons =====
def sufficiency_singletons(df_bin, Ycol, conds):
    rows = []
    y = df_bin[Ycol]

    for c in conds:
        x = df_bin[c]
        inter = np.minimum(x, y).sum()

        rows.append({
            "condition": c,
            "consistency": inter / x.sum() if x.sum() else np.nan,
            "coverage": inter / y.sum() if y.sum() else np.nan,
        })

    return pd.DataFrame(rows).round(4)

# ===== 7) Export results =====
def export_results(tt, keepers, solution, necessity_df, sufficiency_df):
    with pd.ExcelWriter(os.path.join(outdir, "truth_table.xlsx")) as w:
        tt.to_excel(w, index=False)

    with pd.ExcelWriter(os.path.join(outdir, "truth_table_kept.xlsx")) as w:
        keepers.to_excel(w, index=False)

    with pd.ExcelWriter(os.path.join(outdir, "solution.xlsx")) as w:
        pd.DataFrame({"solution": [solution]}).to_excel(w, index=False)

    with pd.ExcelWriter(os.path.join(outdir, "necessity.xlsx")) as w:
        necessity_df.to_excel(w, index=False)

    with pd.ExcelWriter(os.path.join(outdir, "sufficiency.xlsx")) as w:
        sufficiency_df.to_excel(w, index=False)

    print("\nResults saved to:", outdir)

# ===== 8) Plot necessity =====
def plot_necessity(nec_df):
    labels = nec_df["condition"].astype(str).tolist()
    cons = nec_df["consistency"].to_numpy(float)
    cov = nec_df["coverage"].to_numpy(float)

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y - 0.2, cons, height=0.4)
    ax.barh(y + 0.2, cov, height=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "necessity_plot.pdf"))
    plt.show()

# ===== 9) Main execution =====
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Dataset not found: {data_file}")

print("Loading dataset...")
df_bin, Ycol, conds = load_and_binarize(data_file)

print("Building truth table...")
tt, keepers = truth_table_and_keepers(df_bin, Ycol, conds)

print("Minimizing solution...")
solution = minimize_solution(keepers, conds)

print("Running necessity analysis...")
necessity_df = necessity_tables(df_bin, Ycol, conds)

print("Running sufficiency analysis...")
sufficiency_df = sufficiency_singletons(df_bin, Ycol, conds)

export_results(tt, keepers, solution, necessity_df, sufficiency_df)
plot_necessity(necessity_df)

print("\nAll analysis completed.")
