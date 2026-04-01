import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/2分类Y（Y2=心理专注）.csv')

# 划分特征和目标变量
X = df.drop(['Y'], axis=1)
y = df['Y']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=df['Y'])

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

# 定义 XGBoost 二分类模型
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=8)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 定义 K 折交叉验证 (Stratified K-Fold)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)

# 使用网格搜索寻找最佳参数
grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, scoring='accuracy',
                           cv=kfold, verbose=1, n_jobs=-1)

# 拟合模型
grid_search.fit(X_train, y_train)
# 使用最优参数训练模型
xgboost = grid_search.best_estimator_

from sklearn.metrics import classification_report
# 预测测试集
y_pred = xgboost.predict(X_test)
# 输出模型报告，查看评价指标
print(classification_report(y_test, y_pred))

# （1）绘制混淆矩阵热力图1 =================================🩵 🩵 🩵 🩵 🩵 🩵=============================
from sklearn.metrics import confusion_matrix
import seaborn as sns
# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 标准化

# 添加 Total 行和列
cm_with_totals = np.vstack([cm, cm.sum(axis=0)])  # 添加 Total 行
cm_with_totals = np.column_stack([cm_with_totals, cm_with_totals.sum(axis=1)])  # 添加 Total 列

# 创建新的标签（包含 Total）
labels_with_totals = ['malignant', 'benign', 'Total']

# 绘制改进后的混淆矩阵
fig, ax = plt.subplots(figsize=(10, 8))

# 创建颜色映射，设置 Total 行和列为淡灰色
colors = sns.color_palette("Reds", as_cmap=True)
grey_color = "#f0f0f0"  # 淡灰色

# 创建数据用于绘图，Total 行和列保持原始矩阵结构
heatmap_data = cm_normalized.copy()
heatmap_data = np.vstack([heatmap_data, np.zeros((1, heatmap_data.shape[1]))])  # 添加 Total 行
heatmap_data = np.column_stack([heatmap_data, np.zeros((heatmap_data.shape[0], 1))])  # 添加 Total 列

# 绘制主热图
sns.heatmap(
    heatmap_data,
    annot=False,  # 先不添加文字
    fmt="",
    cmap=colors,
    xticklabels=labels_with_totals,
    yticklabels=labels_with_totals,
    cbar=False,
    square=True,
    linewidths=1.5,
    linecolor="white",
    ax=ax,
)

# 覆盖 Total 区域颜色为淡灰色
for i in range(cm.shape[0]):
    ax.add_patch(plt.Rectangle((cm.shape[1], i), 1, 1, fill=True, color=grey_color, edgecolor="white", lw=1.5))
for j in range(cm.shape[1]):
    ax.add_patch(plt.Rectangle((j, cm.shape[0]), 1, 1, fill=True, color=grey_color, edgecolor="white", lw=1.5))
ax.add_patch(plt.Rectangle((cm.shape[1], cm.shape[0]), 1, 1, fill=True, color=grey_color, edgecolor="white", lw=1.5))

# 添加数值和百分比
for i in range(cm_with_totals.shape[0]):
    for j in range(cm_with_totals.shape[1]):
        if i < cm.shape[0] and j < cm.shape[1]:  # 主对角线区域
            value = cm[i, j]
            percentage = cm_normalized[i, j] * 100
            ax.text(j + 0.5, i + 0.5, f"{percentage:.1f}%", ha="center", va="center", fontsize=18, color="black")
            ax.text(j + 0.5, i + 0.65, f"{value}", ha="center", va="center", fontsize=18, color="black")
        elif i == cm.shape[0] or j == cm.shape[1]:  # Total 区域
            total_value = cm_with_totals[i, j]
            if i == cm.shape[0] and j == cm.shape[1]:  # Total 对 Total 的单元格
                ax.text(j + 0.5, i + 0.5, f"{total_value}", ha="center", va="center", fontsize=18, color="black")
            else:  # 非 Total 对 Total 的单元格
                total_percentage = total_value / cm_with_totals[-1, -1] * 100
                ax.text(j + 0.5, i + 0.5, f"{total_percentage:.1f}%", ha="center", va="center", fontsize=18, color="black")
                ax.text(j + 0.5, i + 0.65, f"{total_value}", ha="center", va="center", fontsize=18, color="black")

# 设置标题和轴标签
plt.title(" ", fontsize=20)
plt.xlabel(" ", fontsize=25)
plt.ylabel(" ", fontsize=25)
plt.tight_layout()
plt.savefig("29.6-图1-混淆矩阵热力图（Y2=心理专注）.pdf", format='pdf', bbox_inches='tight')
plt.show()

# （2）绘制图2『TSS概率阈值曲线 』=================================🩵 🩵 🩵 🩵 🩵 🩵=============================
# 获取模型对测试集的预测概率
probabilities = xgboost.predict_proba(X_test)

# 获取真实标签
true_labels = y_test.values
# 初始化列表以存储TSS值
tss_values = []

# 定义阈值范围
thresholds = np.linspace(0, 1, 101)

# 计算不同阈值下的TSS
for threshold in thresholds:
    # 计算预测标签
    predicted_labels = (probabilities[:, 1] > threshold).astype(int)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # 计算灵敏度（Sensitivity）和特异度（Specificity）
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # 计算TSS
    tss = sensitivity + specificity - 1
    tss_values.append(tss)

# 转换TSS值为数组
tss_values = np.array(tss_values)

# 找到最大TSS对应的阈值
max_tss = tss_values.max()
optimal_threshold = thresholds[np.argmax(tss_values)]

# 绘制TSS vs 阈值图
plt.figure(figsize=(8, 6), dpi=1200)
plt.plot(thresholds, tss_values, color='blue', label='Smoothed TSS')
plt.axhline(y=max_tss, color='red', linestyle='--', label=f'Max TSS: {max_tss:.4f}')
plt.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.4f}')

# 在最佳阈值的位置标出交点
plt.scatter(optimal_threshold, max_tss, color='green', zorder=5)
plt.text(optimal_threshold, max_tss, f'({optimal_threshold:.4f}, {max_tss:.4f})',
         color='green', horizontalalignment='left', verticalalignment='bottom')

plt.xlabel(' ', fontsize=25, fontweight='bold')
plt.ylabel(' ', fontsize=25, fontweight='bold')
plt.title(' ', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.savefig("29.6-图2-混淆矩阵热力图（Y2=心理专注）.pdf", format='pdf', bbox_inches='tight')
plt.show()

# （3）绘制图3『混淆矩阵热力』 =================================🩵 🩵 🩵 🩵 🩵 🩵=============================
# 通过数值比较
# 代码简单实现不同预测
# 使用默认阈值（0.5）进行预测
y_pred_default = (probabilities[:, 1] > 0.5).astype(int)

# 使用最优阈值（optimal_threshold）进行预测
y_pred_optimal = (probabilities[:, 1] > optimal_threshold).astype(int)

# 创建 DataFrame 存储真实标签、默认预测值和最优阈值下的预测值
result_df = pd.DataFrame({
    'True_Label': true_labels,
    'y_pred_default': y_pred_default,
    'y_pred_optimal': y_pred_optimal
})

print(classification_report(y_test,  np.array(result_df['y_pred_optimal'])))

# 生成混淆矩阵
cm = confusion_matrix(y_test, np.array(result_df['y_pred_optimal']))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 标准化

# 添加 Total 行和列
cm_with_totals = np.vstack([cm, cm.sum(axis=0)])  # 添加 Total 行
cm_with_totals = np.column_stack([cm_with_totals, cm_with_totals.sum(axis=1)])  # 添加 Total 列

# 创建新的标签（包含 Total）
labels_with_totals = ['malignant', 'benign', 'Total']

# （3）绘制图3『混淆矩阵热力』 =================================🩵 🩵 🩵 🩵 🩵 🩵=============================
# 绘制改进后的混淆矩阵
fig, ax = plt.subplots(figsize=(10, 8))

# 创建颜色映射，设置 Total 行和列为淡灰色
colors = sns.color_palette("Blues", as_cmap=True)
grey_color = "#f0f0f0"  # 淡灰色

# 创建数据用于绘图，Total 行和列保持原始矩阵结构
heatmap_data = cm_normalized.copy()
heatmap_data = np.vstack([heatmap_data, np.zeros((1, heatmap_data.shape[1]))])  # 添加 Total 行
heatmap_data = np.column_stack([heatmap_data, np.zeros((heatmap_data.shape[0], 1))])  # 添加 Total 列

# 绘制主热图
sns.heatmap(
    heatmap_data,
    annot=False,  # 先不添加文字
    fmt="",
    cmap=colors,
    xticklabels=labels_with_totals,
    yticklabels=labels_with_totals,
    cbar=False,
    square=True,
    linewidths=1.5,
    linecolor="white",
    ax=ax,
)

# 覆盖 Total 区域颜色为淡灰色
for i in range(cm.shape[0]):
    ax.add_patch(plt.Rectangle((cm.shape[1], i), 1, 1, fill=True, color=grey_color, edgecolor="white", lw=1.5))
for j in range(cm.shape[1]):
    ax.add_patch(plt.Rectangle((j, cm.shape[0]), 1, 1, fill=True, color=grey_color, edgecolor="white", lw=1.5))
ax.add_patch(plt.Rectangle((cm.shape[1], cm.shape[0]), 1, 1, fill=True, color=grey_color, edgecolor="white", lw=1.5))

# 添加数值和百分比
for i in range(cm_with_totals.shape[0]):
    for j in range(cm_with_totals.shape[1]):
        if i < cm.shape[0] and j < cm.shape[1]:  # 主对角线区域
            value = cm[i, j]
            percentage = cm_normalized[i, j] * 100
            ax.text(j + 0.5, i + 0.5, f"{percentage:.1f}%", ha="center", va="center", fontsize=18, color="black")
            ax.text(j + 0.5, i + 0.65, f"{value}", ha="center", va="center", fontsize=18, color="black")
        elif i == cm.shape[0] or j == cm.shape[1]:  # Total 区域
            total_value = cm_with_totals[i, j]
            if i == cm.shape[0] and j == cm.shape[1]:  # Total 对 Total 的单元格
                ax.text(j + 0.5, i + 0.5, f"{total_value}", ha="center", va="center", fontsize=18, color="black")
            else:  # 非 Total 对 Total 的单元格
                total_percentage = total_value / cm_with_totals[-1, -1] * 100
                ax.text(j + 0.5, i + 0.5, f"{total_percentage:.1f}%", ha="center", va="center", fontsize=18, color="black")
                ax.text(j + 0.5, i + 0.65, f"{total_value}", ha="center", va="center", fontsize=18, color="black")

# 设置标题和轴标签
plt.title(f" ", fontsize=20)
plt.xlabel(" ", fontsize=25)
plt.ylabel(" ", fontsize=25)
plt.tight_layout()
plt.savefig("29.6-图3-混淆矩阵热力图（Y2=心理专注）.pdf", format='pdf', bbox_inches='tight')
plt.show()

# （4）绘制图4『决策树』 =================================🩵 🩵 🩵 🩵 🩵 🩵=============================
import shap
explainer = shap.TreeExplainer(xgboost)
shap_values = explainer.shap_values(X_test)
shap_values_for_first_sample = shap_values[1]  # 从 SHAP 值数组中选择第1个样本的 SHAP 值  python默认从0计位
feature_names = X_test.columns  # 获取 X_test 数据集的列名，即特征名称
original_values = X_test.iloc[1]  # 获取 X_test 中第 1 行的数据作为原始特征值
# 获取 SHAP 解释器的基准值（expected_value）
base_value = explainer.expected_value  # 这是 SHAP 模型的期望值，通常是背景分布的平均值
# 绘制 SHAP 决策图
plt.figure(figsize=(10, 5), dpi=1200)
shap.decision_plot(base_value, shap_values_for_first_sample, original_values, show=False, link='logit')
# plt.savefig("32.1-决策树-图4.pdf", format='pdf', bbox_inches='tight')
plt.tight_layout()
# plt.show()







