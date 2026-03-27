import pandas as pd
import os

# ===== 配置区：根据不同数据集修改这些路径即可 =====
input_path_Y2 = "/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/⑦-20✅top 20（Y2=心理专注）group.csv"

output_dir_Y2 = "/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/🧠大图P3：分组分析结果（Y2=心理专注）"

# ===== 公共的分组函数（保持与Y1一致，确保可比性） =====
def group_pai(pai):
    if pai == 0:
        return 'T1'   # 极低收入
    elif 1 <= pai <= 2:
        return 'T2'   # 低收入
    elif 3 <= pai <= 6:
        return 'T3'   # 中等收入
    else:
        return 'T4'   # 高收入

# ===== 读入Y2数据集 =====
df2 = pd.read_csv(input_path_Y2, encoding='utf-8')

# 生成分组标签
df2['PAI_Group'] = df2['PAI'].apply(group_pai)

# 新建输出文件夹（如果已存在就跳过）
os.makedirs(output_dir_Y2, exist_ok=True)

# 依次输出T1~T4四个csv，并且去掉PAI_Group列
group_names = ['T1', 'T2', 'T3', 'T4']

for g in group_names:
    sub_df2 = df2[df2['PAI_Group'] == g].copy()

    # 删除分组标签列
    if 'PAI_Group' in sub_df2.columns:
        sub_df2 = sub_df2.drop(columns=['PAI_Group'])

    # 构造当前组的输出文件路径
    out_path_g = os.path.join(
        output_dir_Y2,
        f"🧠大图P3：分组分析结果（Y2=心理专注）{g}.csv"
    )

    # 写出到CSV
    sub_df2.to_csv(out_path_g, index=False, encoding='utf-8-sig')

    # 控制台提示
    print(f"{g} 样本量 = {len(sub_df2)}，文件保存至：{out_path_g}")

# 打印一下四组的占比，方便你快速检查分布
print("\n四组占比（Y2=心理专注）:")
print(df2['PAI_Group'].value_counts(normalize=True).round(3))
