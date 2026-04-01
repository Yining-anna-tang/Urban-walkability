# 20.6-浅灰区间学习曲线图（Y1=行为障碍） —— 透明背景、无网格

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.ensemble import RandomForestRegressor

import os

# ======================
# ✅ 2. 路径与数据输入
# ======================
df = pd.read_csv('/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/⑦-18✅top 20（Y2=心理专注）.csv')

X = df.drop(['Y'], axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ======================
# ✅ 模型与学习曲线计算
# ======================
model = RandomForestRegressor(random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train_sizes, train_scores, valid_scores = learning_curve(
    model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 100),
    cv=kf, scoring='neg_mean_squared_error', n_jobs=-1
)

train_scores_mse = -train_scores
valid_scores_mse = -valid_scores

train_mean_mse = np.mean(train_scores_mse, axis=1)
train_std_mse = np.std(train_scores_mse, axis=1)
valid_mean_mse = np.mean(valid_scores_mse, axis=1)
valid_std_mse = np.std(valid_scores_mse, axis=1)

# ======================
# ✅ 准备保存目录
# ======================
save_dir = "/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/🧠补充大图-稳健性检验（Y2=心理专注）"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "20.7-浅灰区间学习曲线图（Y2=心理专注）.pdf")

# ======================
# ✅ 绘图（透明背景、无网格）
# ======================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

fig.patch.set_alpha(0)        # 图背景透明
ax.set_facecolor("none")      # 坐标背景透明
ax.grid(False)                # ❌ 不显示网格

# 训练误差
ax.plot(train_sizes, train_mean_mse,
        linestyle='--', color='black', label='Training Error')

# 验证误差
ax.plot(train_sizes, valid_mean_mse,
        linestyle='-', color='black', label='Validation Error')

# 置信区间
ax.fill_between(train_sizes,
                train_mean_mse - train_std_mse,
                train_mean_mse + train_std_mse,
                alpha=0.2, color='darkgray')

ax.fill_between(train_sizes,
                valid_mean_mse - valid_std_mse,
                valid_mean_mse + valid_std_mse,
                alpha=0.2, color='gray')

# 坐标 & 图例
ax.set_title('', fontsize=14, fontweight='bold')
ax.set_xlabel('', fontsize=12, fontweight='bold')
ax.set_ylabel('', fontsize=12, fontweight='bold')
ax.tick_params(axis='both', labelsize=30)

ax.legend(
    loc='upper center', bbox_to_anchor=(0.5, 1.15),
    ncol=2, fontsize=24, frameon=False
)

ax.set_xlim(left=0)
plt.subplots_adjust(top=0.8)
plt.tight_layout()

# ✅ 保存 PDF 透明背景
plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=1200, transparent=True)
plt.show()

print(f"✅ 图已保存：{save_path}")
