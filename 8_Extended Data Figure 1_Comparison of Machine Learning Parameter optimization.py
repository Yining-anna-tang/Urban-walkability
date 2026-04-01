import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error
from sklearn import metrics
from catboost import CatBoostRegressor
import shap

# ===============================
# 全局绘图与字体设置
# ===============================
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

# ===============================
# 1) 读取数据、设置特征与标签
# ===============================
data = pd.read_csv(
    r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/⑦-18✅top 20（Y2=心理专注）.csv',
    encoding="GBK"
)
df = pd.DataFrame(data)

# 特征与标签
X = df.drop(['Y'], axis=1)
y = df['Y']

# 训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 2) 输出目录（统一放这里）
# ===============================
out_dir = r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning机器学习/学习率调参柱状图（Y2=心理专注）'
os.makedirs(out_dir, exist_ok=True)
print(f'📁 输出目录：{out_dir}')

# 时间戳（用于汇总Excel命名）
ts_all = datetime.now().strftime("%Y%m%d_%H%M%S")

# ===============================
# 3) 学习率循环 & 训练/评估/绘图
# ===============================
learning_rates = [round(x / 100, 2) for x in range(1, 11)]  # 0.01 ~ 0.10
metrics_rows = []  # 用于最终Excel汇总

for lr in learning_rates:
    print("\n" + "="*60)
    print(f"🚀 开始学习率 lr = {lr:.2f} 的训练与评估")
    print("="*60)

    params_cat = {
        'learning_rate': lr,
        'iterations': 1000,
        'depth': 6,
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 500
    }

    # ----- K 折交叉验证，选最佳折的模型 -----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = np.inf
    best_model = None

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = CatBoostRegressor(**params_cat)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)

        y_val_pred = model.predict(X_val)
        score = root_mean_squared_error(y_val, y_val_pred)
        print(f'第 {fold + 1} 折 RMSE: {score:.6f}')

        if score < best_score:
            best_score = score
            best_model = model

    print(f'✅ lr={lr:.2f} 的最佳折 RMSE: {best_score:.6f}')

    # ----- 测试集评估 -----
    y_pred = best_model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    cc = np.corrcoef(y_test, y_pred)[0, 1]
    mean_pred = float(np.mean(y_pred))
    std_pred = float(np.std(y_pred))

    # RSD 安全处理（预测均值接近0时避免除零）
    mean_pred_safe = mean_pred if abs(mean_pred) > 1e-12 else (1e-12 if mean_pred >= 0 else -1e-12)
    rsd = std_pred / mean_pred_safe

    print("\n--- 模型评估指标 ---")
    print(f"RSD: {rsd:.6f}")
    print(f"cc : {cc:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R² : {r2:.6f}")

    # 收集行（保留 6~8 位小数都可，这里统一 6 位）
    metrics_rows.append({
        "learning_rate": lr,
        "RSD": round(rsd, 6),
        "cc": round(float(cc), 6),
        "RMSE": round(rmse, 6),
        "MSE": round(float(mse), 6),
        "MAE": round(float(mae), 6),
        "R_squared": round(float(r2), 6),
        "Best_CV_RMSE": round(float(best_score), 6)  # 记录交叉验证最佳折的RMSE
    })

    # ----- SHAP 特征重要性 & 柱状图 -----
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_ranking = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # 绘图（Top 20）
    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_ranking.head(20),
        palette="viridis"
    )
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()

    lr_str = str(lr).replace('.', 'p')  # 0.01 -> 0p01
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(out_dir, f"📊特征排名条形图_Y2=心理专注_lr{lr_str}_{ts}.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=1200)
    plt.close()
    print(f"🖼️ 已保存图：{pdf_path}")

# ===============================
# 4) 指标汇总到同一张 Excel
# ===============================
metrics_df = pd.DataFrame(metrics_rows).sort_values(by="learning_rate").reset_index(drop=True)

excel_path = os.path.join(out_dir, f"学习率0p01-0p10_模型性能指标汇总_Y2=心理专注_{ts_all}.xlsx")
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    metrics_df.to_excel(writer, index=False, sheet_name="metrics_0.01_0.10")

print("\n🎯 已完成：")
print(f"1) 共输出 10 张特征重要性柱状图（PDF）到：{out_dir}")
print(f"2) 全部学习率的模型性能指标，已汇总在同一张 Excel：{excel_path}")
