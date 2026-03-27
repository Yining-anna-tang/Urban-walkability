import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'Times new Roman'
plt.rcParams['axes.unicode_minus'] = False

# 1.读取数据，设置变量
# 第一处修改：文件路径
data = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/⑥-2✅全样本（Y=E7心理影响专注）.csv',encoding = "GBK")
df = pd.DataFrame(data)
from sklearn.model_selection import train_test_split, KFold

# 第二处修改：检查或修改被解释变量为Y
#
X = df.drop(['Y'],axis=1)
y = df['Y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor

# CatBoost模型参数
params_cat = {
    'learning_rate': 0.01,       # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'iterations': 1000,          # 弱学习器（决策树）的数量
    'depth': 6,                  # 决策树的深度，控制模型复杂度
    'eval_metric': 'RMSE',       # 评估指标，这里使用均方根误差（Root Mean Squared Error，简称RMSE）
    'random_seed': 42,           # 随机种子，用于重现模型的结果
    'verbose': 500               # 控制CatBoost输出信息的详细程度，每100次迭代输出一次
}

# 准备k折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_score = np.inf
best_model = None

# 交叉验证
for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = CatBoostRegressor(**params_cat)
    model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=100)

    # 预测并计算得分
    y_val_pred = model.predict(X_val_fold)
    score = root_mean_squared_error(y_val_fold, y_val_pred)  # RMSE

    scores.append(score)
    print(f'第 {fold + 1} 折 RMSE: {score}')

    # 保存得分最好的模型
    if score < best_score:
        best_score = score
        best_model = model

print(f'Best RMSE: {best_score}')


# 模型评估
from sklearn import metrics

# 预测
y_pred_four = best_model.predict(X_test)

y_pred_list = y_pred_four.tolist()
mse = metrics.mean_squared_error(y_test, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred_list)
r2 = metrics.r2_score(y_test, y_pred_list)

print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("平均绝对误差 (MAE):", mae)
print("拟合优度 (R-squared):", r2)

# 模型解释
# shap解释摘要图
import shap
# 构建 shap解释器
explainer = shap.TreeExplainer(best_model)
# 计算测试集的shap值
shap_values = explainer.shap_values(X_test)
shap_values_numpy = explainer.shap_values(X)
shap_values_Explanation = explainer(X)

# ===============================
# 统一字体设置
# ===============================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16  # 固定字体大小为16

# ===============================
# 📊 批量绘制 SHAP 散点图（顶刊风格）
# ===============================

# ✅ 所有要绘制的特征（根据你提供的最新顺序）
features = [
    'E6',
    '11_depression',
    'E9',
    'E12',
    '3_age',
    'E27a',
    'E27b',
    'E1',
    'E4',
    'E10',
    'E20',
    'E26',
    'E24',
    'E34',
    's3_expectation',
    '2_gender',
    '9_living',
    '7_edu',
    '8_income',
    's2_class mobility'
]

for i, feat in enumerate(features, start=1):
    # --- 绘制 shap 散点图 ---
    shap.dependence_plot(
        feat,
        shap_values_Explanation.values,
        X,
        interaction_index=None,
        show=False,
        dot_size=24   # 散点放大为默认的1.5倍
    )

    # ===== 添加黑色边框 =====
    ax = plt.gca()
    for coll in ax.collections:
        if hasattr(coll, "set_edgecolor"):
            coll.set_edgecolor('black')   # 设置边框颜色
            coll.set_linewidth(0.7)       # 边框线宽（磅）

    # ===== 图形美化 =====
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)

    # 添加 SHAP=0 横线
    plt.axhline(y=0, color='black', linestyle='-.', linewidth=2)

    # 清理文件名中的特殊字符
    safe_feat = feat.replace(' ', '_').replace('/', '_')

    # 保存为 PDF（保持编号与特征名对应）
    plt.savefig(f'环境健康⑤：shap-Y2（心理专注）-{i}（{safe_feat}）.pdf', dpi=1000, format='pdf', bbox_inches='tight')

    plt.show()















