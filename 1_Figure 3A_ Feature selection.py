import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from catboost import CatBoostRegressor
import shap

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ===================== Paths and output directory =====================
base_dir = r'/Users/yiningtang/PycharmProjects/pythonProject1/venv/Machine Learning'
output_dir = os.path.join(base_dir, 'ML_results')
os.makedirs(output_dir, exist_ok=True)

# ===================== Load dataset =====================
data = pd.read_csv(
    os.path.join(base_dir, 'full_sample_Y_E7_psychological_focus.csv'),
    encoding="GBK"
)
df = pd.DataFrame(data)

X = df.drop(['Y'], axis=1)
y = df['Y']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== CatBoost with cross-validation =====================
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
    print(f'Fold {fold + 1} RMSE: {rmse_cv:.6f}')

    if rmse_cv < best_score:
        best_score = rmse_cv
        best_model = model

print(f'Best RMSE (CV): {best_score:.6f}')

# ===================== Test set evaluation (6 metrics) =====================
y_pred = best_model.predict(X_test)
y_pred_list = y_pred.tolist()

mse = metrics.mean_squared_error(y_test, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred_list)
r2 = metrics.r2_score(y_test, y_pred_list)

# Correlation coefficient (cc)
cc = np.corrcoef(y_test.values.flatten(), np.array(y_pred_list).flatten())[0, 1]

# Relative standard deviation (RSD)
# Defined as the standard deviation of residuals divided by the mean of true values
residuals = y_test.values.flatten() - np.array(y_pred_list).flatten()
rsd = np.std(residuals, ddof=1) / (np.mean(y_test.values) if np.mean(y_test.values) != 0 else 1)

print("\n===== Test Set Evaluation Metrics =====")
print(f"1. Relative Standard Deviation (RSD): {rsd:.12f}")
print(f"2. Correlation Coefficient (cc): {cc:.12f}")
print(f"3. Root Mean Squared Error (RMSE): {rmse:.12f}")
print(f"4. Mean Squared Error (MSE): {mse:.12f}")
print(f"5. Mean Absolute Error (MAE): {mae:.12f}")
print(f"6. R-squared (R²): {r2:.12f}")

# Save evaluation metrics to CSV
metrics_df = pd.DataFrame({
    'metric': ['RSD', 'cc', 'RMSE', 'MSE', 'MAE', 'R2'],
    'value': [rsd, cc, rmse, mse, mae, r2]
})
metrics_path = os.path.join(output_dir, 'model_evaluation_metrics_Y2.csv')
metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')

# ===================== Model interpretation using SHAP =====================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Set font configuration for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 15

max_n = min(500, X_test.shape[0])

# Construct SHAP Explanation object for heatmap plotting
shap_explanation = shap.Explanation(
    values=shap_values[:max_n, :],
    base_values=explainer.expected_value,
    data=X_test.iloc[:max_n, :],
    feature_names=X_test.columns
)

# ============== Figure 1: SHAP summary plot (dot) ==============
plt.figure()
shap.summary_plot(shap_values[:max_n, :], X_test.iloc[:max_n, :], show=False)
plt.tight_layout()
fig1_path = os.path.join(output_dir, 'SHAP_summary_dot_Y2.pdf')
plt.savefig(fig1_path, dpi=600, format='pdf')
plt.close()

# ============== Figure 2: SHAP heatmap ==============
plt.figure()
shap.plots.heatmap(shap_explanation, show=False)
plt.tight_layout()
fig2_path = os.path.join(output_dir, 'SHAP_heatmap_Y2.pdf')
plt.savefig(fig2_path, dpi=600, format='pdf')
plt.close()

# ============== Figure 3: SHAP summary plot (bar) ==============
plt.figure()
shap.summary_plot(shap_values[:max_n, :], X_test.iloc[:max_n, :], plot_type='bar', show=False)
plt.tight_layout()
fig3_path = os.path.join(output_dir, 'SHAP_summary_bar_Y2.pdf')
plt.savefig(fig3_path, dpi=600, format='pdf')
plt.close()

# ============== Figure 4: SHAP interaction summary plot ==============
# Interaction computation is resource-intensive; reduce max_n for faster execution if needed
plt.figure()
shap_interaction_values = explainer.shap_interaction_values(X_test.iloc[:max_n, :])
shap.summary_plot(shap_interaction_values, X_test.iloc[:max_n, :], show=False)
plt.tight_layout()
fig4_path = os.path.join(output_dir, 'SHAP_interaction_summary_Y2.pdf')
plt.savefig(fig4_path, dpi=600, format='pdf')
plt.close()

print(f"\nAll figures and metric files have been saved to: {output_dir}")
