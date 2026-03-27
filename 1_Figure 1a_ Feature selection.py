import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  # ← 新增
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from catboost import CatBoostRegressor
import shap

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径与输出文件夹 =====================
base_dir = r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习'
output_dir = os.path.join(base_dir, '🧠ML结果：Y2=心理专注')  # ← 新增
os.makedirs(output_dir, exist_ok=True)  # ← 新增

# ===================== 导入数据集 =====================
data = pd.read_csv(
    os.path.join(base_dir, '⑥-2✅全样本（Y=E7心理影响专注）.csv'),
    encoding="GBK"
)
df = pd.DataFrame(data)

X = df.drop(['Y'], axis=1)
y = df['Y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== CatBoost & 交叉验证 =====================
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
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=100
    )

    y_val_pred = model.predict(X_val_fold)
    rmse_cv = np.sqrt(metrics.mean_squared_error(y_val_fold, y_val_pred))
    scores.append(rmse_cv)
    print(f'第 {fold + 1} 折 RMSE: {rmse_cv:.6f}')

    if rmse_cv < best_score:
        best_score = rmse_cv
        best_model = model

print(f'Best RMSE (CV): {best_score:.6f}')

# ===================== 测试集评估（6个指标） =====================
y_pred = best_model.predict(X_test)
y_pred_list = y_pred.tolist()

mse = metrics.mean_squared_error(y_test, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred_list)
r2 = metrics.r2_score(y_test, y_pred_list)

# 相关系数（cc）
cc = np.corrcoef(y_test.values.flatten(), np.array(y_pred_list).flatten())[0, 1]

# 相对标准差（RSD）
# 这里按“残差的标准差 / 真实值均值”定义；如你有自定义口径，可替换为你需要的公式
residuals = y_test.values.flatten() - np.array(y_pred_list).flatten()
rsd = np.std(residuals, ddof=1) / (np.mean(y_test.values) if np.mean(y_test.values) != 0 else 1)

print("\n===== 测试集 6 指标 =====")
print(f"1. 相对标准差 (RSD): {rsd:.12f}")
print(f"2. 相关系数 (cc): {cc:.12f}")
print(f"3. 均方根误差 (RMSE): {rmse:.12f}")
print(f"4. 均方误差 (MSE): {mse:.12f}")
print(f"5. 平均绝对误差 (MAE): {mae:.12f}")
print(f"6. 拟合优度 (R-squared): {r2:.12f}")

# 保存指标为 CSV
metrics_df = pd.DataFrame({
    'metric': ['RSD', 'cc', 'RMSE', 'MSE', 'MAE', 'R2'],
    'value': [rsd, cc, rmse, mse, mae, r2]
})
metrics_path = os.path.join(output_dir, '模型评估指标（Y2=心理专注）.csv')
metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')

# ===================== 模型解释（SHAP） =====================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# 统一字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimHei']
plt.rcParams['font.size'] = 15

max_n = min(500, X_test.shape[0])

# 构造 Explanation（用于热力图）
shap_explanation = shap.Explanation(
    values=shap_values[:max_n, :],
    base_values=explainer.expected_value,
    data=X_test.iloc[:max_n, :],
    feature_names=X_test.columns
)

# ============== 图1) SHAP 摘要图（散点） ==============
plt.figure()
shap.summary_plot(shap_values[:max_n, :], X_test.iloc[:max_n, :], show=False)
plt.tight_layout()
fig1_path = os.path.join(output_dir, '①SHAP摘要图-散点（Y2=心理专注）.pdf')
plt.savefig(fig1_path, dpi=600, format='pdf')
plt.close()

# ============== 图2) SHAP 热力图 ==============
plt.figure()
shap.plots.heatmap(shap_explanation, show=False)
plt.tight_layout()
fig2_path = os.path.join(output_dir, '②SHAP热力图（Y2=心理专注）.pdf')
plt.savefig(fig2_path, dpi=600, format='pdf')
plt.close()

# ============== 图3) SHAP 摘要图（条形） ==============
plt.figure()
shap.summary_plot(shap_values[:max_n, :], X_test.iloc[:max_n, :], plot_type='bar', show=False)
plt.tight_layout()
fig3_path = os.path.join(output_dir, '③SHAP摘要图-条形（Y2=心理专注）.pdf')
plt.savefig(fig3_path, dpi=600, format='pdf')
plt.close()

# ============== 图4) SHAP 交互作用摘要图 ==============
# 交互作用计算量较大，如需更快可进一步下采样 max_n
plt.figure()
shap_interaction_values = explainer.shap_interaction_values(X_test.iloc[:max_n, :])
shap.summary_plot(shap_interaction_values, X_test.iloc[:max_n, :], show=False)
plt.tight_layout()
fig4_path = os.path.join(output_dir, '④SHAP交互作用摘要图（Y2=心理专注）.pdf')
plt.savefig(fig4_path, dpi=600, format='pdf')
plt.close()

print(f"\nPDF 图与指标文件已保存至：{output_dir}")
