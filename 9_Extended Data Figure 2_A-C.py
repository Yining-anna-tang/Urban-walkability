import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
df = pd.read_csv(r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/⑦-18✅top 20（Y2=心理专注）.csv', encoding ="GBK")
from sklearn.model_selection import train_test_split

# 划分特征和目标变量
X = df.drop(['Y'], axis=1)
y = df['Y']
print(X.shape)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化存储结果的DataFrame
selection_results = pd.DataFrame(columns=['Feature', 'Importance', 'MSE', 'R2'])

# 创建随机森林回归器
rf_reg = RandomForestRegressor(random_state=42)  # ① 创建随机森林回归器：每次消除一个特征

# 使用递归特征消除（RFE）方法，结合随机森林回归模型对模拟数据集进行特征的筛选
# 目的是通过逐步消除不重要的特征来找到最能预测“Tensile strength/MPa”的特征
# 使用递归特征消除（RFE）
rfe = RFE(estimator=rf_reg, n_features_to_select=1, step=1)  # ② 挑特征：每次消除一个特征
rfe.fit(X_train, y_train)  # ③ 现在开始训练：X_train是所有特征的集合；y_train是希望预测的目标Y；RFE会带着随机森林一遍遍地看这些特征，每次淘汰一个最没用的，直到找到那个对预测结果最有帮助的特征。

# 获取特征排名
feature_ranking = rfe.ranking_

# 构建特征排名表
rfe_features = pd.DataFrame({
    'Feature': X_train.columns,
    'Ranking': feature_ranking
}).sort_values(by='Ranking')
rfe_features

# ↓ ↓ ↓
# 初始化用于训练的特征列表
selected_features = []

# 根据RFE排序依次选择特征
for i in range(len(rfe_features)):
    # 当前特征
    current_feature = rfe_features.iloc[i]['Feature']
    selected_features.append(current_feature)

    # 训练模型（仅使用当前选定的特征）
    X_train_subset = X_train[selected_features]
    X_test_subset = X_test[selected_features]

    # 创建并训练随机森林回归模型
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train_subset, y_train)

    # 获取特征重要性（随机森林使用 feature_importances_）
    importance = rf_reg.feature_importances_[len(selected_features) - 1]

    # 预测并计算均方误差（MSE）和R²分数
    y_pred = rf_reg.predict(X_test_subset)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 保存结果
    selection_results.loc[len(selection_results)] = [
        current_feature,
        importance,
        mse,
        r2
    ]
selection_results


# P1（R²）：📈 📈 📈 📈 📈 📈拟合优度趋于1的可视化效果 📈 📈 📈 📈 📈 📈
# 简介：可视化递归特征消除过程，展示特征重要性与模型性能（R²和MSE）的变化，可以看出，当选择到第11个特征时，拟合优度R²达到了较高值并趋于平缓
# ↓ ↓ ↓
selection_results = selection_results.iloc[0:30]  # 可视化前30个特征
# 参数：选择前 n 个特征
n_features = 12
fig, ax1 = plt.subplots(figsize=(16, 6))
# 渐变柱状图：特征贡献度
norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
colors = plt.cm.Blues(norm(selection_results['Importance']))
ax1.bar(selection_results['Feature'], selection_results['Importance'], color=colors, label=' ')
ax1.set_xlabel(" ", fontsize=18, fontweight='bold')
ax1.set_ylabel(" ", fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelsize=15, width=1.5)  # 设置y轴刻度大小为15
x_labels = selection_results['Feature']
x_colors = ['red' if i < n_features else 'black' for i in range(len(x_labels))]
for tick_label, color in zip(ax1.get_xticklabels(), x_colors):
    tick_label.set_color(color)
ax1.tick_params(axis='x', rotation=75, labelsize=15, width=1.5)  # 设置x轴刻度大小为15
ax2 = ax1.twinx()
ax2.plot(
    selection_results['Feature'][:n_features + 1],  # 连接红点到黑点的过渡
    selection_results['R2'][:n_features + 1],
    color="red", marker='o', linestyle='-', label=" "
)
ax2.plot(
    selection_results['Feature'][n_features:],
    selection_results['R2'][n_features:],
    color="black", marker='o', linestyle='-', label=" "
)
ax2.set_ylabel(" ", fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelsize=15, width=1.5)  # 设置y轴刻度大小为15
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))  # 保留三位小数
# 添加标题和图例
plt.title(f" ", fontsize=18, fontweight='bold')
fig.tight_layout()
# 📌📌📌📌📌📌📌📌📌
plt.savefig("13.6-图1-递归法R2指标代码（Y2=心理专注）.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()





# P2（MSE趋势图）：📉 📉 📉 📉 📉 📉均方误差趋于0的可视化效果 📉 📉 📉 📉 📉 📉
# 简介：可视化递归特征消除过程，展示模型性能（均方误差：MSE）的变化，可以看出，当选择到第11个特征时，均方误差MSE达到了达到较小值并趋于平缓
# ↓ ↓ ↓
fig, ax1 = plt.subplots(figsize=(16, 6))

# 渐变柱状图：特征贡献度
norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
colors = plt.cm.Blues(norm(selection_results['Importance']))
ax1.bar(selection_results['Feature'], selection_results['Importance'], color=colors, label='Feature Importance')
ax1.set_xlabel(" ", fontsize=18, fontweight='bold')
ax1.set_ylabel(" ", fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelsize=15, width=1.5)  # 设置y轴刻度大小为15

# 修改 x 轴特征颜色，前 n_features 用红色，其他用黑色
x_labels = selection_results['Feature']
x_colors = ['red' if i < n_features else 'black' for i in range(len(x_labels))]
for tick_label, color in zip(ax1.get_xticklabels(), x_colors):
    tick_label.set_color(color)
ax1.tick_params(axis='x', rotation=75, labelsize=15, width=1.5)  # 设置x轴刻度大小为15

# 创建第二个y轴
ax2 = ax1.twinx()

# 折线图：MSE成绩（Cumulative MSE）
ax2.plot(
    selection_results['Feature'][:n_features + 1],  # 连接红点到黑点的过渡
    selection_results['MSE'][:n_features + 1],
    color="red", marker='o', linestyle='-', label=" "
)

# 黑点和黑线：其余特征的MSE
ax2.plot(
    selection_results['Feature'][n_features:],
    selection_results['MSE'][n_features:],
    color="black", marker='o', linestyle='-', label=" "
)

# 设置y轴标签
ax2.set_ylabel(" ", fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelsize=15, width=1.5)  # 设置y轴刻度大小为15
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))  # 保留三位小数

# 添加标题和图例
plt.title(f" ", fontsize=18, fontweight='bold')

# 调整布局
fig.tight_layout()

# 保存图表
# 📌📌📌
plt.savefig("13.6-图2-递归法R2指标代码（Y2=心理专注）.pdf", format='pdf', bbox_inches='tight', dpi=1200)
# 显示图表
plt.show()


# 🐍 🐍 🐍 🐍 🐍 🐍利用递归特征得到最优子子集重新建模 🐍 🐍 🐍 🐍 🐍 🐍
selection_results.iloc[0:11]['Feature']

from sklearn.model_selection import train_test_split
# 划分特征和目标变量
X = df[selection_results.iloc[0:11]['Feature']]
y = df['Y']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import KFold
# 定义随机森林回归模型
rf_model = RandomForestRegressor(random_state=42)

# 定义网格搜索的参数空间（包括默认参数）
param_grid = {
    'n_estimators': [100, 200],  # 树的数量
    'max_depth': [None, 10, 20],  # 树的最大深度
    'min_samples_split': [2, 5],  # 分割一个内部节点所需的最小样本数
    'min_samples_leaf': [1, 2],  # 叶子节点的最小样本数
    'bootstrap': [True, False]  # 是否使用bootstrap样本
}

# 使用 GridSearchCV 进行网格搜索，采用 KFold 进行 K 折交叉验证
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=KFold(n_splits=5),
                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# 在训练集上进行网格搜索
grid_search.fit(X_train, y_train)

# 输出网格搜索的最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数训练模型
best_rf_model = grid_search.best_estimator_

# 问题行此处开始：❎ ❎ ❎ ❎ ❎ ❎
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 用训练好的模型在训练集上进行预测
y_pred_train = best_rf_model.predict(X_train)

# 计算 RMSE (Root Mean Squared Error) 对训练集
rmse_train = mean_squared_error(y_train, y_pred_train)
# 计算 MAE (Mean Absolute Error) 对训练集
mae_train = mean_absolute_error(y_train, y_pred_train)
# 计算 R² (R-squared) 对训练集
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集的评估指标
print(f"RMSE on Train Set: {rmse_train}")
print(f"MAE on Train Set: {mae_train}")
print(f"R² on Train Set: {r2_train}")

# 用训练好的模型在测试集上进行预测
y_pred_test = best_rf_model.predict(X_test)

# 计算 RMSE (Root Mean Squared Error) 对测试集
rmse_test = mean_squared_error(y_test, y_pred_test)
# 计算 MAE (Mean Absolute Error) 对测试集
mae_test = mean_absolute_error(y_test, y_pred_test)
# 计算 R² (R-squared) 对测试集
r2_test = r2_score(y_test, y_pred_test)

# 输出测试集的评估指标
print(f"RMSE on Test Set: {rmse_test}")
print(f"MAE on Test Set: {mae_test}")
print(f"R² on Test Set: {r2_test}")

# e.g.
# RMSE on Train Set: 25.202753385221854
# MAE on Train Set: 16.950277876443707
# R² on Train Set: 0.9685898480540663
# RMSE on Test Set: 47.69748446411317
# MAE on Test Set: 34.2818202871461
# R² on Test Set: 0.9178513662586597

# P3（评估指标可视化）：🐍 🐍 🐍 🐍 🐍 🐍评估指标可视化展示 🐍 🐍 🐍 🐍 🐍 🐍
# 计算残差 (Residuals)
residuals = y_test - y_pred_test

# 创建正方形画布
fig, ax = plt.subplots(figsize=(8, 6), dpi=1200)

# 绘制测试集散点，颜色由残差决定
scatter = ax.scatter(y_test, y_pred_test, c=residuals, cmap='jet', s=30)  # 使用'jet'渐变色表示残差

# 设置坐标轴范围
x_min, x_max = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
ax.set_xlim([x_min, x_max])
ax.set_ylim([x_min, x_max])

# 添加 x=y 参考线（红色虚线），延伸整个画布
ax.plot([x_min, x_max], [x_min, x_max], 'r--', linewidth=1, label='y = x line')

# 计算回归线（线性拟合）
slope, intercept = np.polyfit(y_test, y_pred_test, 1)
regression_line = slope * np.array([x_min, x_max]) + intercept

# 绘制回归线，延伸整个画布
ax.plot([x_min, x_max], regression_line, color='black', linewidth=1, label='Regression Line')

# 计算预测带的上下限 (95% Prediction Band)
std_err = np.std(residuals)
pred_upper = regression_line + 1.96 * std_err
pred_lower = regression_line - 1.96 * std_err

# 绘制预测带的上下限，延伸整个画布
ax.plot([x_min, x_max], pred_upper, 'k--', linewidth=0.8, label='95% Prediction Band')
ax.plot([x_min, x_max], pred_lower, 'k--', linewidth=0.8)

# 设置标签
ax.set_xlabel(' ', fontsize=15)
ax.set_ylabel(' ', fontsize=15)

# 添加网格线
ax.grid(True, linestyle='--', linewidth=0.8, color='#DCDCDC')

# 图例框
ax.legend(loc='upper left', fontsize=15, frameon=False)

# 显示测试集评价指标在右下角
test_metrics_text = f"RMSE = {rmse_test:.3f}\n MAE = {mae_test:.3f}\nR² = {r2_test:.3f}"
ax.text(0.95, 0.05, f"{test_metrics_text}", transform=ax.transAxes, fontsize=15,
        verticalalignment='bottom', horizontalalignment='right', color='black')

# 显示颜色条（色标）
cbar = plt.colorbar(scatter, ax=ax, label=' ')  # 创建颜色条对象
cbar.set_ticks([np.min(residuals), 0, np.max(residuals)])  # 设置颜色条刻度
cbar.ax.tick_params(labelsize=12)  # 设置颜色条刻度字体大小
cbar.set_label(' ', fontsize=15)  # 设置标签字体大小
# 📌📌📌
plt.savefig("13.6-图3（Y2=心理专注）.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

# 🐍 🐍 🐍 🐍 🐍 🐍步骤（一）🐍 🐍 🐍 🐍 🐍 🐍
# 使用SHAP的TreeExplainer对训练好的随机森林模型进行解释，并计算测试集的SHAP值
# 为接下来的基于SHAP和GAM识别特征影响阈值的模型解释和可视化做准备
# ① 解释模型：
import shap
explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
shap_values_df.head()

X_test = X_test.reset_index(drop=True)

# P4（SHAP图）：🤗 🤗 🤗 🤗 🤗特征贡献的SHAP可视化🤗 🤗 🤗 🤗 🤗
# 简介：先可视化一个特征（# 📌📌📌WPWD）的SHAP值，展示该特征对模型输出的影响
from pygam import LinearGAM, s


# ② 构建 GAM 模型，选取一个特征：
print(X_test.columns)
# 📌📌📌
X = X_test['LSI'].values.reshape(-1, 1)  # GAM 输入必须是二维
y = shap_values_df['LSI'].values

# 调整 n_splines 和 lam 以更好地拟合数据，避免警告或获得更平滑的曲线
# 如果数据点较少，可以减少 n_splines
gam = LinearGAM(s(0, n_splines=min(10, len(X)-1), lam=0.6)).fit(X, y) # 单变量平滑

# 生成预测值和置信区间
XX = gam.generate_X_grid(term=0, n=200)  # X 的网格点, 增加 n 使曲线更平滑
y_pred = gam.predict(XX)  # GAM 拟合的预测值
confidence_interval = gam.prediction_intervals(XX, width=0.95)  # 置信区间

# 提取拟合优度 R²
R2 = gam.statistics_['pseudo_r2']['explained_deviance']  # 提取 explained_deviance
R2 = round(R2, 3)  # 保留三位小数

# 找到 y=0 与拟合线的所有交点 (更精确的查找)
tipping_points_x = []
tipping_points_y = []

# 查找从负到正的交叉点
for i in range(len(XX.flatten()) - 1):
    if y_pred[i] < 0 and y_pred[i+1] > 0:
        x1, y1_val = XX.flatten()[i], y_pred[i]
        x2, y2_val = XX.flatten()[i+1], y_pred[i+1]
        if y2_val - y1_val != 0:  # 避免除以零
            tipping_points_x.append(x1 - y1_val * (x2 - x1) / (y2_val - y1_val))
            tipping_points_y.append(0)  # 理论上交点在 y=0

# 如果没有从负到正的，尝试找从正到负的交叉点
if not tipping_points_x:
    for i in range(len(XX.flatten()) - 1):
        if y_pred[i] > 0 and y_pred[i+1] < 0:
            x1, y1_val = XX.flatten()[i], y_pred[i]
            x2, y2_val = XX.flatten()[i+1], y_pred[i+1]
            if y2_val - y1_val != 0:  # 避免除以零
                tipping_points_x.append(x1 - y1_val * (x2 - x1) / (y2_val - y1_val))
                tipping_points_y.append(0)  # 理论上交点在 y=0

# 绘制图像
plt.figure(figsize=(6, 4.5), dpi=300)  # 调整figsize和dpi以获得更好的输出
ax = plt.gca()  # 获取当前坐标轴

# 添加参考线 y=0（灰色虚线）
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=0)

# 绘制置信区间（填充区域，颜色 #D3D3D3，透明度 0.7）
plt.fill_between(
    XX.flatten(),
    confidence_interval[:, 0],  # 下置信边界
    confidence_interval[:, 1],  # 上置信边界
    color="#C0C0C0", alpha=0.6, label="95% CI", zorder=1  # 稍微深一点的灰色，根据示例图调整
)

# 绘制趋势线（实线，颜色，线宽参考示例图）
plt.plot(XX.flatten(), y_pred, color="#A52A2A", linewidth=2.5, zorder=2, label="GAM fit")  # 棕色/暗红色

# *** 添加背景颜色填充 ***
# 获取当前y轴的范围，以便axhspan能正确填充
y_min_plot, y_max_plot = ax.get_ylim()

# y > 0 部分的背景 (浅薄荷绿色)
ax.axhspan(0, y_max_plot, facecolor='xkcd:light mint green', alpha=0.3, zorder=-1)  # zorder=-1确保在最底层

# y < 0 部分的背景 (浅橙色/桃色)
ax.axhspan(y_min_plot, 0, facecolor='xkcd:pale orange', alpha=0.3, zorder=-1)
# 重新设置y轴范围，以防axhspan改变了它
ax.set_ylim(y_min_plot, y_max_plot)

# 标记所有交点
for tipping_point_x_value in tipping_points_x:
    ax.annotate('Tipping point',
                xy=(tipping_point_x_value, 0),  # 箭头指向的位置
                xytext=(tipping_point_x_value - (XX.flatten().max() - XX.flatten().min())*0.1, -max(abs(y_min_plot), abs(y_max_plot))*0.2),  # 文本位置，根据图调整
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='black'),
                fontsize=12, color="black"
               )

# P 值通过 gam.statistics_ 中获取
P_value = gam.statistics_['p_values'][0]  # 假设第一个变量的 P 值
# 根据 P 值范围设置显示内容
if P_value < 0.001:
    P_text = r"$\it{p} < 0.001$"  # 使用斜体
elif P_value < 0.01:
    P_text = r"$\it{p} < 0.01$"
elif P_value < 0.05:
    P_text = r"$\it{p} < 0.05$"
else:
    P_text = f"$\it{{p}} = {P_value:.3f}$"  # 保留三位小数，斜体

# 添加拟合优度，并在其下方显示 P 值描述
annotation_text = f"$R^2 = {R2:.2f}$\n{P_text}"  # R2保留两位小数
plt.text(
    0.05, 0.95,  # 坐标位置（相对于左上角）
    annotation_text,
    transform=ax.transAxes,  # 使用轴的相对坐标
    fontsize=10,
    verticalalignment='top',
    color="black",
    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.6)  # 可选：给文本添加背景框
)

plt.xlabel(' ', fontsize=11)  # 动态获取特征名称
plt.ylabel(' ', fontsize=11)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
# 📌📌📌
plt.savefig("13.6-图4（Y2=心理专注）-单个特征-LSI.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()