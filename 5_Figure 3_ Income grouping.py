import pandas as pd
import os

# ===== Configuration section: modify these paths for different datasets =====
input_path_Y2 = "/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning/machine_learning/07-20_top20_(Y2=mental_focus)_group.csv"

output_dir_Y2 = "/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning/machine_learning/Figure_P3_grouped_results_(Y2=mental_focus)"

# ===== Common grouping function (kept identical to Y1 to ensure comparability) =====
def group_pai(pai):
    if pai == 0:
        return 'T1'   # Extremely low income
    elif 1 <= pai <= 2:
        return 'T2'   # Low income
    elif 3 <= pai <= 6:
        return 'T3'   # Middle income
    else:
        return 'T4'   # High income

# ===== Read Y2 dataset =====
df2 = pd.read_csv(input_path_Y2, encoding='utf-8')

# Generate group labels
df2['PAI_Group'] = df2['PAI'].apply(group_pai)

# Create output folder (skip if it already exists)
os.makedirs(output_dir_Y2, exist_ok=True)

# Export T1–T4 four CSV files and remove PAI_Group column
group_names = ['T1', 'T2', 'T3', 'T4']

for g in group_names:
    sub_df2 = df2[df2['PAI_Group'] == g].copy()

    # Remove grouping label column
    if 'PAI_Group' in sub_df2.columns:
        sub_df2 = sub_df2.drop(columns=['PAI_Group'])

    # Construct output file path for current group
    out_path_g = os.path.join(
        output_dir_Y2,
        f"Figure_P3_grouped_results_(Y2=mental_focus)_{g}.csv"
    )

    # Write to CSV
    sub_df2.to_csv(out_path_g, index=False, encoding='utf-8-sig')

    # Console message
    print(f"{g} sample size = {len(sub_df2)}, file saved to: {out_path_g}")

# Print group proportions for quick distribution check
print("\nGroup proportions (Y2 = mental focus):")
print(df2['PAI_Group'].value_counts(normalize=True).round(3))
