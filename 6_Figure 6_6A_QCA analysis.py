# -*- coding: utf-8 -*-
# 批量 fsQCA（0/1 二值版）循环：t1~t4
# pip install pandas sympy openpyxl

import pandas as pd
import numpy as np
from pathlib import Path
from sympy import symbols
from sympy.logic.boolalg import SOPform, simplify_logic

# ===== 0) 公共参数 =====
# 一致性 & 频数阈值（充分性筛选）
N_CUT   = 1
INCL_CUT = 0.80

# ===== 1) 输入与输出路径 =====
root = Path("/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/Y2：QCA结果")

# 要处理的数据文件（按需要增减）
data_files = [
    root / "t1（0-1）.csv",
    root / "t2（0-1）.csv",
    root / "t3（0-1）.csv",
    root / "t4（0-1）.csv",
]

# 输出目录（与用户要求相同）
outdir = root
outdir.mkdir(exist_ok=True)

def load_and_binarize(csv_path):
    """读取 CSV，自动识别 Y 列，并将所有列二值化为 0/1（>0 记为 1）"""
    df = pd.read_csv(csv_path, encoding="utf-8")
    outcome_candidates = [c for c in df.columns if c.lower() in ("y","outcome","result")]
    if not outcome_candidates:
        raise ValueError(f"{csv_path.name} 未发现结果列，请命名为 y / Y / outcome / result 之一。")
    Ycol = outcome_candidates[0]
    conds = [c for c in df.columns if c != Ycol]

    df_bin = df.copy()
    for c in [*conds, Ycol]:
        df_bin[c] = (pd.to_numeric(df_bin[c], errors="coerce").fillna(0) > 0).astype(int)
    return df_bin, Ycol, conds

def truth_table_and_keepers(df_bin, Ycol, conds, n_cut=N_CUT, incl_cut=INCL_CUT):
    """真值表 + 通过阈值的配置（充分性指标：配置→Y）"""
    g = df_bin.groupby(conds, dropna=False)
    tt = g[Ycol].agg(n="count", n_suff="sum").reset_index()
    tt["consistency"] = tt["n_suff"] / tt["n"].replace(0, np.nan)
    total_y1 = df_bin[Ycol].sum()
    tt["coverage"] = tt["n_suff"] / (total_y1 if total_y1 else 1)
    keepers = tt.query("n >= @n_cut and consistency >= @incl_cut").copy()
    return tt, keepers

def minimize_solution(keepers, conds):
    """对保留配置做 SOP 最小化；若无保留配置则返回提示字符串"""
    mins = keepers[conds].values.tolist()
    if not mins:
        return "No solution under current thresholds."
    sym_vars = symbols(" ".join(conds))
    expr = SOPform(sym_vars, mins, [])
    expr_simplified = simplify_logic(expr, form='dnf')
    return str(expr_simplified)

def necessity_tables(df_bin, Ycol, conds):
    """必要性：对 X 与 ~X 计算一致性与覆盖率；并按 A..F, ~A..~F 或原列顺序输出"""
    def necessity_one(x, y, name):
        x = pd.to_numeric(x, errors="coerce").fillna(0).clip(0,1).values
        y = pd.to_numeric(y, errors="coerce").fillna(0).clip(0,1).values
        inter = np.minimum(x, y).sum()
        sum_y = y.sum(); sum_x = x.sum()
        cons = inter / sum_y if sum_y > 0 else np.nan
        cov  = inter / sum_x if sum_x > 0 else np.nan
        return {"condition": name, "type": "X",
                "sum_X": int(sum_x), "sum_Y": int(sum_y), "intersection": int(inter),
                "consistency_necessity": cons, "coverage_necessity": cov}

    def necessity_neg_one(x, y, name):
        xnot = 1 - pd.to_numeric(x, errors="coerce").fillna(0).clip(0,1).values
        y    = pd.to_numeric(y, errors="coerce").fillna(0).clip(0,1).values
        inter = np.minimum(xnot, y).sum()
        sum_y = y.sum(); sum_xnot = xnot.sum()
        cons = inter / sum_y if sum_y > 0 else np.nan
        cov  = inter / sum_xnot if sum_xnot > 0 else np.nan
        return {"condition": f"~{name}", "type": "~X",
                "sum_X": int(sum_xnot), "sum_Y": int(sum_y), "intersection": int(inter),
                "consistency_necessity": cons, "coverage_necessity": cov}

    yvals = df_bin[Ycol]
    nec_rows = []
    for c in conds:
        nec_rows.append(necessity_one(df_bin[c], yvals, c))
        nec_rows.append(necessity_neg_one(df_bin[c], yvals, c))
    necessity_df = pd.DataFrame(nec_rows)

    # 固定顺序：若刚好是 A..F，则强制 A..F；否则按原列顺序
    target_order = conds.copy()
    if set(conds) == set(list("ABCDEF")) and len(conds) == 6:
        target_order = list("ABCDEF")
    ordered_conditions = target_order + [f"~{c}" for c in target_order]
    cat = pd.CategoricalDtype(categories=ordered_conditions, ordered=True)
    necessity_df["condition"] = necessity_df["condition"].astype(cat)
    necessity_df = necessity_df.sort_values("condition")

    # 四位小数
    for col in ["consistency_necessity", "coverage_necessity"]:
        necessity_df[col] = necessity_df[col].astype(float).round(4)

    return necessity_df

def sufficiency_singletons(df_bin, Ycol, conds):
    """单变量充分性（X→Y），按 A..F 或原列顺序输出"""
    def sufficiency_one(x, y, name):
        x = pd.to_numeric(x, errors="coerce").fillna(0).clip(0,1).values
        y = pd.to_numeric(y, errors="coerce").fillna(0).clip(0,1).values
        inter = np.minimum(x, y).sum()
        sum_x = x.sum(); sum_y = y.sum()
        cons = inter / sum_x if sum_x > 0 else np.nan
        cov  = inter / sum_y if sum_y > 0 else np.nan
        return {"condition": name, "sum_X": int(sum_x), "sum_Y": int(sum_y), "intersection": int(inter),
                "consistency_sufficiency": cons, "coverage_sufficiency": cov}

    rows = [sufficiency_one(df_bin[c], df_bin[Ycol], c) for c in conds]
    suff_df = pd.DataFrame(rows)

    target_order = conds.copy()
    if set(conds) == set(list("ABCDEF")) and len(conds) == 6:
        target_order = list("ABCDEF")
    cat = pd.CategoricalDtype(categories=target_order, ordered=True)
    suff_df["condition"] = suff_df["condition"].astype(cat)
    suff_df = suff_df.sort_values("condition")

    for col in ["consistency_sufficiency", "coverage_sufficiency"]:
        suff_df[col] = suff_df[col].astype(float).round(4)

    return suff_df

def export_all(tt, keepers, solution_str, necessity_df, sufficiency_df, out_prefix: Path):
    """将 5 个结果导出为 .xlsx，保留 4 位小数并使用给定前缀（含数据集名）"""
    # 真值表四位小数
    tt_out = tt.copy()
    if "consistency" in tt_out: tt_out["consistency"] = tt_out["consistency"].astype(float).round(4)
    if "coverage"    in tt_out: tt_out["coverage"]    = tt_out["coverage"].astype(float).round(4)

    keepers_out = keepers.copy()
    if not keepers_out.empty:
        if "consistency" in keepers_out: keepers_out["consistency"] = keepers_out["consistency"].astype(float).round(4)
        if "coverage"    in keepers_out: keepers_out["coverage"]    = keepers_out["coverage"].astype(float).round(4)

    # 文件名（带数据集前缀）
    file1 = out_prefix.parent / f"{out_prefix.name} - ① truth_table.xlsx"
    file2 = out_prefix.parent / f"{out_prefix.name} - ② truth_table_kept.xlsx"
    file3 = out_prefix.parent / f"{out_prefix.name} - ③ solution.xlsx"
    file4 = out_prefix.parent / f"{out_prefix.name} - ④ necessity.xlsx"
    file5 = out_prefix.parent / f"{out_prefix.name} - ⑤ sufficiency.xlsx"

    with pd.ExcelWriter(file1, engine="openpyxl") as w:
        tt_out.to_excel(w, index=False, sheet_name="truth_table")

    with pd.ExcelWriter(file2, engine="openpyxl") as w:
        (keepers_out if not keepers_out.empty else pd.DataFrame()).to_excel(
            w, index=False, sheet_name="kept_configs"
        )

    with pd.ExcelWriter(file3, engine="openpyxl") as w:
        pd.DataFrame({"solution_sop": [solution_str]}).to_excel(w, index=False, sheet_name="solution")

    with pd.ExcelWriter(file4, engine="openpyxl") as w:
        necessity_df.to_excel(w, index=False, sheet_name="necessity")

    with pd.ExcelWriter(file5, engine="openpyxl") as w:
        sufficiency_df.to_excel(w, index=False, sheet_name="sufficiency")

    print(f"\n✅ 结果已保存到：{out_prefix.parent}")
    print(f"📂 {file1.name}")
    print(f"📂 {file2.name}")
    print(f"📄 {file3.name}")
    print(f"📂 {file4.name}   （必要性：按 A..F, ~A..~F 排列）")
    print(f"📂 {file5.name}   （充分性：按 A..F 排列）")

# ===== 2) 主循环 =====
for csv_path in data_files:
    if not csv_path.exists():
        print(f"⚠️ 跳过：文件不存在 -> {csv_path}")
        continue

    # 数据集前缀（如 t1 / t2 / t3 / t4）
    prefix = csv_path.stem  # 例如 "t1（0-1）"
    out_prefix = outdir / prefix

    print(f"\n====== 开始处理：{csv_path.name} ======")
    df_bin, Ycol, conds = load_and_binarize(csv_path)

    # 真值表 & 保留配置
    tt, keepers = truth_table_and_keepers(df_bin, Ycol, conds,
                                          n_cut=N_CUT, incl_cut=INCL_CUT)

    # 最小化解
    solution_str = minimize_solution(keepers, conds)

    # 必要性 & 单变量充分性
    necessity_df   = necessity_tables(df_bin, Ycol, conds)
    sufficiency_df = sufficiency_singletons(df_bin, Ycol, conds)

    # 导出
    export_all(tt, keepers, solution_str, necessity_df, sufficiency_df, out_prefix)

print("\n🎉 全部数据集处理完成。")

# ===== 绘图：必要性（Consistency / Coverage）横向分组柱状图，PDF导出 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 若你的脚本上面已定义 outdir，这里直接使用；否则给一个兜底路径
if "outdir" not in globals():
    outdir = Path("/Users/yiningtang/PycharmProjects/pythonProject1/venv/\
Machine Learning机器学习/Y1：QCA结果")

# 全局字体：Arial, 30号（保持你的设置）
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
})

def _order_conditions_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """按 A..F, ~A..~F 固定顺序；若不是 A..F，则按表中正项顺序 + 否定"""
    df = df.copy()
    need_cols = ["condition", "type", "consistency_necessity", "coverage_necessity"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"必要性表缺少列：{missing}")

    pos_list = df.loc[df["type"].astype(str).str.strip().eq("X"),
                      "condition"].astype(str).tolist()
    if set(pos_list) == set(list("ABCDEF")) and len(pos_list) == 6:
        target_order = list("ABCDEF")
    else:
        target_order = pos_list

    ordered = target_order + [f"~{c}" for c in target_order]
    cat = pd.CategoricalDtype(categories=ordered, ordered=True)
    df["condition"] = df["condition"].astype(str).astype(cat)
    df = df.sort_values("condition")

    for col in ["consistency_necessity", "coverage_necessity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    return df

# 需要绘图的 necessity.xlsx 清单
nec_files = [
    outdir / "t1（0-1） - ④ necessity.xlsx",
    outdir / "t2（0-1） - ④ necessity.xlsx",
    outdir / "t3（0-1） - ④ necessity.xlsx",
    outdir / "t4（0-1） - ④ necessity.xlsx",
]

for nec_path in nec_files:
    if not nec_path.exists():
        print(f"⚠️ 未找到必要性文件，跳过：{nec_path}")
        continue

    nec = pd.read_excel(nec_path, sheet_name="necessity")
    try:
        nec = _order_conditions_for_plot(nec)
    except Exception as e:
        print(f"⚠️ {nec_path.name} 排序或列检查失败：{e}")
        continue

    labels = nec["condition"].astype(str).tolist()
    cons = nec["consistency_necessity"].to_numpy(float)
    covg = nec["coverage_necessity"].to_numpy(float)

    y = np.arange(len(labels))
    # —— 布局与间距参数（避免重叠）——
    h   = 0.56         # 每根柱子的厚度（略粗）
    off = h * 0.70     # 两根柱心距的一半；> h/2，保证有缝隙不重叠

    # 随条目数量自适应画布高度（避免拥挤）
    fig_h = max(7, 0.70 * len(labels) + 1.5)

    # 透明背景：figure 和 axes 都设透明
    fig = plt.figure(figsize=(11, fig_h), facecolor='none')
    ax = fig.add_subplot(111, facecolor='none')

    # 红蓝配色 + 你当前的 50% 透明度 + 黑色边框（0.75 pt）
    ax.barh(
        y - off, cons, height=h,
        color="#D32F2F", edgecolor='black', linewidth=0.75, alpha=0.5
    )
    ax.barh(
        y + off, covg, height=h,
        color="#1565C0", edgecolor='black', linewidth=0.75, alpha=0.5
    )

    # y 轴标签与取值范围
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)

    # 删网格、删图例、删主标题、删 x 轴标题（保持之前要求）
    ax.grid(False)
    if ax.get_legend():
        ax.legend().remove()
    ax.set_title("")
    ax.set_xlabel("")

    # 去外框，仅保留坐标轴线（下、左）
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)

    # 左侧留更多空隙，避免长标签被裁剪
    plt.subplots_adjust(left=0.25, right=0.98, top=0.98, bottom=0.06)
    fig.tight_layout()

    # 透明背景保存为 PDF
    pdf_path = nec_path.with_name(f"{nec_path.stem} - 必要性条形图.pdf")
    fig.savefig(pdf_path, format="pdf", transparent=True)
    print(f"✅ 已保存图像：{pdf_path}")

    # 展示
    plt.show()
    plt.close(fig)





