# 导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 设置 Matplotlib 的默认字体为 Arial，避免乱码
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 导入训练集测试集拆分工具
from sklearn.model_selection import train_test_split

# 读取数据集，假设文件路径为 '2025-2-20公众号Python机器学习AI.xlsx'
df = pd.read_csv('/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/2分类Y（Y2=心理专注）.csv')

# 查看数据的前几行，了解数据结构
df.head()

# 划分特征和目标变量
X = df.drop(['Y'], axis=1)
y = df['Y']

# 划分训练集和测试集
# train_test_split 会随机划分数据集，test_size=0.3 表示 30% 的数据用于测试集，70% 用于训练集
# random_state=42 保证结果可重复
# stratify=df['Y'] 保证训练集和测试集中的目标变量比例一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=df['Y'])

# 图(A)复现 =================================📉 📉 📉 📉 📉 📉=============================
import lightgbm as lgb

# 创建LGBM分类器
lgbm_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
# 训练模型
lgbm_clf.fit(X_train, y_train)

# 获取特征重要性
feature_importances = lgbm_clf.feature_importances_

lgbm_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

lgbm_feature_importance

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 假设 lgbm_feature_importance 已经按照 Importance 排序
top_features = lgbm_feature_importance.sort_values(by='Importance', ascending=False)

# 初始化存储结果的DataFrame
selection_A_LGBM = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])
selection_A_LogReg = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])
selection_A_RF = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])
selection_A_XGB = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])

# 初始化用于训练的特征列表
selected_features = []

# K折交叉验证
kf = KFold(n_splits=4, shuffle=True, random_state=42)
n_splits = kf.get_n_splits()

# 动态创建列名
fold_columns = [f'Fold_{i+1}_ROC' for i in range(n_splits)]

# 依次添加特征
for i in range(len(top_features)):
    current_feature = top_features.iloc[i]['Feature']
    selected_features.append(current_feature)

    fold_roc_scores_LGBM = []
    fold_roc_scores_LogReg = []
    fold_roc_scores_RF = []
    fold_roc_scores_XGB = []

    # K折交叉验证
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx][selected_features], X_train.iloc[val_idx][selected_features]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # LightGBM 分类模型
        lgbm_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
        lgbm_clf.fit(X_train_fold, y_train_fold)
        y_val_proba = lgbm_clf.predict_proba(X_val_fold)[:, 1]
        fold_roc_scores_LGBM.append(roc_auc_score(y_val_fold, y_val_proba))

        # Logistic Regression 需要数据标准化
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        logreg_clf = LogisticRegression(random_state=42, max_iter=200)
        logreg_clf.fit(X_train_fold_scaled, y_train_fold)
        y_val_proba = logreg_clf.predict_proba(X_val_fold_scaled)[:, 1]
        fold_roc_scores_LogReg.append(roc_auc_score(y_val_fold, y_val_proba))

        # 随机森林分类器
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(X_train_fold, y_train_fold)
        y_val_proba = rf_clf.predict_proba(X_val_fold)[:, 1]
        fold_roc_scores_RF.append(roc_auc_score(y_val_fold, y_val_proba))

        # XGBoost 分类器
        xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_clf.fit(X_train_fold, y_train_fold)
        y_val_proba = xgb_clf.predict_proba(X_val_fold)[:, 1]
        fold_roc_scores_XGB.append(roc_auc_score(y_val_fold, y_val_proba))

    # 计算平均ROC AUC分数
    mean_roc_score_LGBM = np.mean(fold_roc_scores_LGBM)
    mean_roc_score_LogReg = np.mean(fold_roc_scores_LogReg)
    mean_roc_score_RF = np.mean(fold_roc_scores_RF)
    mean_roc_score_XGB = np.mean(fold_roc_scores_XGB)

    # 保存LGBM结果
    row_data_LGBM = {'Feature': current_feature, 'Importance': top_features.iloc[i]['Importance'], 'Mean_ROC': mean_roc_score_LGBM}
    for j, score in enumerate(fold_roc_scores_LGBM):
        row_data_LGBM[fold_columns[j]] = score
    selection_A_LGBM = pd.concat([selection_A_LGBM, pd.DataFrame([row_data_LGBM])], ignore_index=True)

    # 保存LogReg结果
    row_data_LogReg = {'Feature': current_feature, 'Importance': top_features.iloc[i]['Importance'], 'Mean_ROC': mean_roc_score_LogReg}
    for j, score in enumerate(fold_roc_scores_LogReg):
        row_data_LogReg[fold_columns[j]] = score
    selection_A_LogReg = pd.concat([selection_A_LogReg, pd.DataFrame([row_data_LogReg])], ignore_index=True)

    # 保存RF结果
    row_data_RF = {'Feature': current_feature, 'Importance': top_features.iloc[i]['Importance'], 'Mean_ROC': mean_roc_score_RF}
    for j, score in enumerate(fold_roc_scores_RF):
        row_data_RF[fold_columns[j]] = score
    selection_A_RF = pd.concat([selection_A_RF, pd.DataFrame([row_data_RF])], ignore_index=True)

    # 保存XGB结果
    row_data_XGB = {'Feature': current_feature, 'Importance': top_features.iloc[i]['Importance'], 'Mean_ROC': mean_roc_score_XGB}
    for j, score in enumerate(fold_roc_scores_XGB):
        row_data_XGB[fold_columns[j]] = score
    selection_A_XGB = pd.concat([selection_A_XGB, pd.DataFrame([row_data_XGB])], ignore_index=True)

selection_A_LGBM, selection_A_LogReg, selection_A_RF, selection_A_XGB

# 生成图：
from matplotlib.lines import Line2D

# 为每个模型绘制AUC分数随特征数量变化的曲线
def plot_auc_per_feature(selection_A, label, color, ax):
    # 生成特征数量的顺序，从1开始
    feature_count = np.arange(1, len(selection_A) + 1)

    # 计算标准误差
    std_error = selection_A.iloc[:, 3:].std(axis=1) / np.sqrt(len(selection_A.iloc[:, 3:]))

    # 计算95%置信区间的误差条 (1.96 * 标准误差)
    yerr = 1.96 * std_error

    ax.errorbar(feature_count, selection_A['Mean_ROC'], yerr=yerr,
                label=label, color=color, capsize=5, alpha=0.7)  # 设置透明度

    ax.set_xlabel('Number of features', fontsize=14)  # 增大X轴标签字体
    ax.set_ylabel('AUC', fontsize=14)  # 增大Y轴标签字体
    ax.set_xticklabels([str(int(x)) for x in ax.get_xticks()], fontsize=12)  # 增大X轴刻度字体
    ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontsize=12)  # 增大Y轴刻度字体，保留两位小数

    # 关闭顶部和右边的轴
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # 设置x轴的刻度，从0开始，但0位置没有数据
    xticks = np.arange(0, len(selection_A) + 1, 5)  # 从0到最大特征数，每隔5个显示一次
    ax.set_xticks(xticks)  # 设置x轴的刻度

    # 自动调整y轴的范围，根据数据的最大最小值
    ax.set_ylim([0.3, 1])  # 设置Y轴范围

# 创建子图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制不同模型的曲线
plot_auc_per_feature(selection_A_LGBM, 'Light GBM', 'orange', ax)
plot_auc_per_feature(selection_A_LogReg, 'Logistic Regression', 'green', ax)
plot_auc_per_feature(selection_A_RF, 'Random Forest', 'red', ax)
plot_auc_per_feature(selection_A_XGB, 'XGBoost', 'blue', ax)

# 自定义图例为横线，不显示圆点
legend_lines = [
    Line2D([0], [0], color='orange', lw=2),   # Light GBM
    Line2D([0], [0], color='green', lw=2),    # Logistic Regression
    Line2D([0], [0], color='red', lw=2),      # Random Forest
    Line2D([0], [0], color='blue', lw=2)      # XGBoost
]
ax.legend(handles=legend_lines,
          labels=['Light GBM', 'Logistic Regression', 'Random Forest', 'XGBoost'],
          loc='lower right', fontsize=14)  # 设置图例位置为右下角

# 保存并显示图形
plt.savefig("28.6-最佳特征数量（Y2=心理专注）.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()