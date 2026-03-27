# ===============================
# 导入基础库
# ===============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error
from sklearn import metrics
from catboost import CatBoostRegressor

# ===============================
# 全局字体设置：Arial
# ===============================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ===============================
# 读取数据并划分
# ============= 📌📌📌📌 1.修改数据集路径 📌📌📌📌 ==================
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/⑦-18✅top 20（Y2=心理专注）.csv', encoding="GBK")
df = pd.DataFrame(data)
X = df.drop(['Y'], axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# CatBoost模型
# ===============================
# ============= 📌📌📌📌 2.修改模型参数 📌📌📌📌 ==================
params_cat = {
    'learning_rate': 0.02,
    'iterations': 1000,
    'depth': 6,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 500
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_score = np.inf
best_model = None

for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    model = CatBoostRegressor(**params_cat)
    model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=100)
    y_val_pred = model.predict(X_val_fold)
    score = root_mean_squared_error(y_val_fold, y_val_pred)
    scores.append(score)
    print(f'第 {fold + 1} 折 RMSE: {score}')
    if score < best_score:
        best_score = score
        best_model = model

print(f'Best RMSE: {best_score}')

# ===============================
# 模型评估
# ===============================
y_pred = best_model.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print("1.均方误差 (MSE):", mse)
print("2.均方根误差 (RMSE):", rmse)
print("3.平均绝对误差 (MAE):", mae)
print("4.拟合优度 (R-squared):", r2)

# ===============================
# SHAP解释
# ===============================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# ===============================
# 📊 批量绘制 SHAP 交互依赖图（interaction = s2_class mobility）
# ===============================
# ============= 📌📌📌📌 3.修改特征变量 📌📌📌📌 ==================
# ========= 特征列表 =========
# ========= 特征列表 =========
all_features = [
    'EBD',
    'EI',
    'DMF',
    'PHL',
    'SRH',
    'LSI',
    'GEN',
    'DV',
    'ST',
    'AGE',
    'SD',
    'HFA',
    'MUF',
    'EPK',
    'UWHF',
    'EDU',
    'PSM',
    'LSC',
    'PAI'
]
interaction_feat = 'HSAA'
features = [f for f in all_features if f != interaction_feat]

# ============= 📌📌📌📌 4.修改输出路径 📌📌📌📌 ==================
out_dir = "🧠交互作用2D PDP图汇总（Y2=心理专注）"
os.makedirs(out_dir, exist_ok=True)

for i, feat in enumerate(features, start=1):
    plt.figure(figsize=(6, 4))
    shap.dependence_plot(
        feat, shap_values, X_test,
        interaction_index=interaction_feat,
        show=False,
        dot_size=24
    )

    ax = plt.gca()
    # ===== 设置边框 =====
    for coll in ax.collections:
        if hasattr(coll, "set_edgecolor"):
            coll.set_edgecolor('black')
            coll.set_linewidth(0.7)

    # ===== 字体 & 样式 =====
    plt.xticks(fontsize=21, fontfamily='Arial')
    plt.yticks(fontsize=21, fontfamily='Arial')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # ===== 删除右侧颜色条标题 & 修改字体 =====
    cb_ax = plt.gcf().axes[-1]  # 获取颜色条轴
    cb_ax.set_ylabel('')  # 删除颜色条标题
    cb_ax.tick_params(labelsize=21)  # 设置字体大小
    for label in cb_ax.get_yticklabels():
        label.set_fontfamily('Arial')

    # ===== 保存 =====
    # ============= 📌📌📌📌 5.修改输出路径 📌📌📌📌 ==================
    def safe(s): return s.replace(' ', '_').replace('/', '_')
    fname = f"{out_dir}/环境健康⑥-PDP（Y2=心理专注）-{i:02d}（{safe(feat)}__int_{safe(interaction_feat)}）.pdf"
    plt.savefig(fname, format='pdf', bbox_inches='tight', dpi=1200)
    print("Saved:", fname)
    plt.close()
