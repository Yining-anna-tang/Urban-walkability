# ===============================
# Import core libraries
# ===============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error as root_mean_squared_error
from sklearn import metrics
from catboost import CatBoostRegressor

# ===============================
# Global font settings: Arial
# ===============================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ===============================
# Load dataset and split
# ===============================
# Update dataset path
data = pd.read_csv(r'1_Dataset.csv', encoding="GBK")
df = pd.DataFrame(data)
X = df.drop(['Y'], axis=1)
y = df['Y']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# CatBoost model
# ===============================
# Update model parameters
params_cat = {
    'learning_rate': 0.02,
    'iterations': 1000,
    'depth': 6,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 500
}

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_score = np.inf
best_model = None

for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = CatBoostRegressor(**params_cat)
    model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=100)
    
    y_val_pred = model.predict(X_val_fold)
    score = root_mean_squared_error(y_val_fold, y_val_pred)
    scores.append(score)
    
    print(f'Fold {fold + 1} RMSE: {score}')
    
    if score < best_score:
        best_score = score
        best_model = model

print(f'Best RMSE: {best_score}')

# ===============================
# Model evaluation
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
# SHAP explanation
# ===============================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# ===============================
# 📊 Batch plot SHAP 2D interaction plots (interaction = HSAA)
# ===============================
all_features = [
    'EBD', 'EI', 'DMF', 'PHL', 'SRH', 'LSI', 'GEN', 'DV', 'ST', 'AGE',
    'SD', 'HFA', 'MUF', 'EPK', 'UWHF', 'EDU', 'PSM', 'LSC', 'PAI'
]
interaction_feat = 'HSAA'
features = [f for f in all_features if f != interaction_feat]

# Update output directory
out_dir = "Brain_Interaction_2D_PDP_Y2_PsychFocus"
os.makedirs(out_dir, exist_ok=True)

for i, feat in enumerate(features, start=1):
    plt.figure(figsize=(6, 4))
    shap.dependence_plot(
        feat, shap_values, X_test,
        interaction_index=interaction_feat,
        show=False,
        dot_size=24
    )

    ax = plt.gca()
    # ===== Add black borders to points =====
    for coll in ax.collections:
        if hasattr(coll, "set_edgecolor"):
            coll.set_edgecolor('black')
            coll.set_linewidth(0.7)

    # ===== Font & style =====
    plt.xticks(fontsize=21, fontfamily='Arial')
    plt.yticks(fontsize=21, fontfamily='Arial')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # ===== Remove colorbar label & set font =====
    cb_ax = plt.gcf().axes[-1]
    cb_ax.set_ylabel('')
    cb_ax.tick_params(labelsize=21)
    for label in cb_ax.get_yticklabels():
        label.set_fontfamily('Arial')

    # ===== Save plot =====
    def safe(s): return s.replace(' ', '_').replace('/', '_')
    fname = f"{out_dir}/PDP_Y2_PsychFocus-{i:02d}({safe(feat)}__int_{safe(interaction_feat)}).pdf"
    plt.savefig(fname, format='pdf', bbox_inches='tight', dpi=1200)
    print("Saved:", fname)
    plt.close()
