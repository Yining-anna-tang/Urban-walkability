# ===============================
# Import libraries
# ===============================
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error as root_mean_squared_error
from sklearn import metrics
from catboost import CatBoostRegressor
from sklearn.inspection import PartialDependenceDisplay

# ===============================
# Global font & symbol settings (Arial)
# ===============================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign displays correctly

# ===============================
# Load dataset and split
# ===============================
# Update dataset path
data = pd.read_csv(r'1_Dataset.csv', encoding="GBK")
df = pd.DataFrame(data)
X = df.drop(['Y'], axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# CatBoost model & 5-Fold cross-validation
# ===============================
params_cat = {
    'learning_rate': 0.02,
    'iterations': 1000,
    'depth': 6,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 500
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = np.inf
best_model = None

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    model = CatBoostRegressor(**params_cat)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)

    y_va_pred = model.predict(X_va)
    score = root_mean_squared_error(y_va, y_va_pred)
    print(f'Fold {fold} RMSE: {score:.6f}')

    if score < best_score:
        best_score = score
        best_model = model

print(f'Best RMSE: {best_score:.6f}')

# ===============================
# Evaluate on test set
# ===============================
y_pred = best_model.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("1. Mean Squared Error (MSE):", mse)
print("2. Root Mean Squared Error (RMSE):", rmse)
print("3. Mean Absolute Error (MAE):", mae)
print("4. R-squared:", r2)

# ===============================
# SHAP explainer and shap_values
# ===============================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Optional SHAP explanation snippet for downstream use
shap_explanation = shap.Explanation(
    values=shap_values[0:500, :],
    base_values=explainer.expected_value,
    data=X_test.iloc[0:500, :],
    feature_names=X_test.columns
)

# ===============================
# Output directory
# ===============================
out_dir = r'Brain_Interaction_Seaweed_Plots_Y2_PsychFocus'
os.makedirs(out_dir, exist_ok=True)

# ===============================
# Features (including interaction_feat for 2D PDP)
# ===============================
all_features = [
    'EBD', 'EI', 'DMF', 'PHL', 'SRH', 'LSI', 'GEN', 'DV', 'ST', 'AGE',
    'SD', 'HFA'
]
interaction_feat = 'HSAA'

# ===============================
# Loop to plot 3 types per feature:
# Plot1: Average PDP; Plot2: ICE; Plot3: 2D PDP (feat × interaction_feat)
# ===============================
for idx, feat in enumerate(all_features, start=1):
    # ---------- Plot1: Average PDP ----------
    plt.figure(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        best_model,
        X_test,
        features=[feat],
        kind='average',
        grid_resolution=50
    )
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=21, fontfamily='Arial')
    plt.yticks(fontsize=21, fontfamily='Arial')

    fname1 = os.path.join(out_dir, f"SeaweedPlot_Y2-Plot1_AvgEffect_{feat}.pdf")
    plt.savefig(fname1, format='pdf', bbox_inches='tight', dpi=1200)
    print(f"Saved: {fname1}")
    plt.close()

    # ---------- Plot2: ICE ----------
    plt.figure(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        best_model,
        X_test,
        features=[feat],
        kind='individual',
        grid_resolution=50
    )
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=21, fontfamily='Arial')
    plt.yticks(fontsize=21, fontfamily='Arial')

    fname2 = os.path.join(out_dir, f"SeaweedPlot_Y2-Plot2_IndividualEffect_{feat}.pdf")
    plt.savefig(fname2, format='pdf', bbox_inches='tight', dpi=1200)
    print(f"Saved: {fname2}")
    plt.close()

    # ---------- Plot3: 2D PDP (feat × interaction_feat) ----------
    plt.figure(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        best_model,
        X_test,
        features=[[feat, interaction_feat]],
        kind='average',
        grid_resolution=50,
        contour_kw={'cmap': 'viridis', 'alpha': 0.8}
    )
    plt.suptitle('')  # Remove title
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=21, fontfamily='Arial')
    plt.yticks(fontsize=21, fontfamily='Arial')

    fname3 = os.path.join(out_dir, f"SeaweedPlot_Y2-Plot3_2D_PDP_{feat}__{interaction_feat}.pdf")
    plt.savefig(fname3, format='pdf', bbox_inches='tight', dpi=1200)
    print(f"Saved: {fname3}")
    plt.close()
