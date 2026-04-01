import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
df = pd.read_csv('/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/2分类Y（Y2=心理专注）.csv')

from sklearn.model_selection import train_test_split

# 划分特征和目标变量
X = df.drop(['Y'], axis=1)
y = df['Y']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=df['Y']
)

import xgboost as xgb
from sklearn.metrics import accuracy_score

# 初始化XGBoost分类器
model = xgb.XGBClassifier(random_state=42)

# 训练模型
model.fit(X_train, y_train)

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
shap_values_df.head()

# 📌📌📌📌📌📌 根据新备注分类变量重写

# ① 连续特征（4个）
continuous_features = ['LSC', 'AGE', 'PAI', 'EDU']
continuous_df = shap_values_df[continuous_features]

# ② 分类型特征（6个）
categorical_features = ['EBD', 'LSI', 'HSAA', 'EI', 'DMF', 'EPK']

categorical_df = shap_values_df[categorical_features]


from statsmodels.nonparametric.smoothers_lowess import lowess

# 合并 SHAP 特征（1连续 + 5分类）
features = continuous_df.columns.tolist() + categorical_df.columns.tolist()

# 创建图形和子图
fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 创建2×5=10个子图
axes = axes.flatten()

# 循环绘制每个特征的散点图
for i in range(len(axes)):
    if i < len(features):  # 如果还有特征未绘制
        feature = features[i]
        if feature in X_test.columns and feature in shap_values_df.columns:
            ax = axes[i]

            # 绘制散点图，颜色基于特征的原始值
            scatter = ax.scatter(X_test[feature], shap_values_df[feature], s=30, c=X_test[feature],
                                 cmap='coolwarm', edgecolor='k')

            # 添加横线
            ax.axhline(y=0, color='red', linestyle='-.', linewidth=1)

            # LOWESS 拟合
            lowess_fit = lowess(shap_values_df[feature], X_test[feature], frac=0.3)
            ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], color='#B5B5B5', linewidth=2)

            # 标签设置
            ax.set_xlabel(feature, fontsize=18)
            ax.set_ylabel('', fontsize=18)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('', fontsize=16)
        else:
            axes[i].axis('off')
    else:
        axes[i].axis('off')

# 子图间距与保存设置
plt.subplots_adjust(hspace=0.4, wspace=1.0)
plt.savefig("18.6-SHAP散点图（Y2=心理专注）10子图.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.tight_layout()
plt.show()