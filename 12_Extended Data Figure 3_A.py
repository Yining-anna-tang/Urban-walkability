import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyALE import ale

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

df = pd.read_csv('/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/⑦-18✅top 20（Y2=心理专注）.csv')


from sklearn.model_selection import train_test_split

# 划分特征和目标变量
X = df.drop(['Y'],axis=1)
y = df['Y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=df['Y']
)

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# 初始化随机森林模型
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# 初始化Boruta特征选择器
boruta_selector = BorutaPy(
    rf,                           # 使用的基础分类器，这里是一个随机森林模型
    n_estimators='auto',          # 树的数量，这里设置为'auto'，Boruta会自动确定适当的树的数量
    verbose=2,                     # 显示详细的日志信息，'2'表示打印详细的过程信息
    random_state=42                # 随机数种子，确保实验可重现
)

# 对训练数据进行特征选择
boruta_selector.fit(X_train.values, y_train.values)
# 检查选中的特征
selected_features = X_train.columns[boruta_selector.support_].to_list()
# 打印被选择的特征
print("Selected Features: ", selected_features)
# 打印被剔除的特征
rejected_features = X_train.columns[~boruta_selector.support_].to_list()
print("Rejected Features: ", rejected_features)
# 打印有待定性的特征
tentative_features = X_train.columns[boruta_selector.support_weak_].to_list()
print("Tentative Features: ", tentative_features)

# 初始化随机森林模型
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# 初始化存储特征排名的 DataFrame
ranking_df = pd.DataFrame(index=range(1, 21), columns=X_train.columns)
# 运行 Boruta 20 次
for i in range(20):
    print(f"Iteration {i + 1}")

    # 初始化Boruta特征选择器
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=i, max_iter=50)

    # 对训练数据进行特征选择
    boruta_selector.fit(X_train.values, y_train.values)

    # 获取特征排名
    feature_ranks = boruta_selector.ranking_

    # 将特征排名保存到 DataFrame 中
    ranking_df.loc[i + 1] = feature_ranks

# 绘图：
import seaborn as sns

# 确保数据集中只有数值列
numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')

# 计算每个特征的中位数
median_values = numeric_ranking_df.median()

# 根据中位数对列进行排序
sorted_columns = median_values.sort_values().index

# 获取Boruta选择的特征、剔除的特征和待定的特征
selected_features = X_train.columns[boruta_selector.support_].to_list()
rejected_features = X_train.columns[~boruta_selector.support_].to_list()
tentative_features = X_train.columns[boruta_selector.support_weak_].to_list()

# 创建颜色映射字典
color_map = {feature: "#02BBC1" for feature in selected_features}  # 选中特征颜色
color_map.update({feature: "#E53935" for feature in rejected_features})  # 被拒绝特征颜色
color_map.update({feature: "#FFC107" for feature in tentative_features})  # 待定特征颜色

# 设置绘图风格
plt.figure(figsize=(15, 8))
sns.set(style="whitegrid")

# 绘制箱线图
ax = sns.boxplot(data=numeric_ranking_df[sorted_columns], palette=color_map)

# 设置x轴标签的旋转角度
plt.xticks(rotation=90)

# 添加标题和标签
plt.title("Sorted Feature Ranking Distribution by Boruta", fontsize=20, fontweight='bold')  # 增大标题字体
plt.xlabel("Attributes", fontsize=18, fontweight='bold')  # 增大x轴标签字体
plt.ylabel("Importance ranking", fontsize=18, fontweight='bold')  # 增大y轴标签字体

# 设置 x 轴标签的颜色与图例相同
for tick, label in zip(range(len(sorted_columns)), ax.get_xticklabels()):
    feature = sorted_columns[tick]
    label.set_color(color_map.get(feature, "black"))  # 直接从 color_map 获取颜色

# 增加 x 轴标签字体大小
for label in ax.get_xticklabels():
    label.set_fontsize(16)  # 设置x轴标签字体大小

# 增大坐标轴刻度字体
ax.tick_params(axis='both', which='major', labelsize=16)

# 添加图例
handles = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#02BBC1", markersize=15, label='Selected'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#E53935", markersize=15, label='Rejected'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#FFC107", markersize=15, label='Tentative')
]
plt.legend(handles=handles, title="Feature Status", loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title_fontsize=18, fontsize=16, fancybox=True, shadow=True, labelspacing=1.5)

# 调整布局
plt.tight_layout()
plt.savefig("27.6-长X轴的柱状图（特征摘要排序）Y2=心理专注.pdf", format='pdf', bbox_inches='tight',dpi=1200)
plt.show()

