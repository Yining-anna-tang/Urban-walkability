import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 1. Load dataset and set variables
# Update file path
data = pd.read_csv(r'1_Dataset.csv', encoding="GBK")
df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split, KFold

# Check or modify dependent variable as Y
X = df.drop(['Y'], axis=1)
y = df['Y']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import mean_squared_error as root_mean_squared_error
from catboost import CatBoostRegressor

# CatBoost model parameters
params_cat = {
    'learning_rate': 0.01,       # Learning rate: step size, prevents overfitting, typical range: 0.01 - 0.1
    'iterations': 1000,          # Number of weak learners (decision trees)
    'depth': 6,                  # Depth of trees, controls model complexity
    'eval_metric': 'RMSE',       # Evaluation metric: Root Mean Squared Error
    'random_seed': 42,           # Random seed for reproducibility
    'verbose': 500               # Control output frequency: every 500 iterations
}

# Prepare K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_score = np.inf
best_model = None

# Cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = CatBoostRegressor(**params_cat)
    model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=100)

    # Predict and calculate score
    y_val_pred = model.predict(X_val_fold)
    score = root_mean_squared_error(y_val_fold, y_val_pred)

    scores.append(score)
    print(f'Fold {fold + 1} RMSE: {score}')

    # Save the best model
    if score < best_score:
        best_score = score
        best_model = model

print(f'Best RMSE: {best_score}')

# Model evaluation
from sklearn import metrics

# Predict on test set
y_pred = best_model.predict(X_test)
y_pred_list = y_pred.tolist()

mse = metrics.mean_squared_error(y_test, y_pred_list)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, y_pred_list)
r2 = metrics.r2_score(y_test, y_pred_list)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared:", r2)

# Model interpretation with SHAP
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(best_model)

# Compute SHAP values for test and full dataset
shap_values = explainer.shap_values(X_test)
shap_values_numpy = explainer.shap_values(X)
shap_values_Explanation = explainer(X)

# ===============================
# Unified font settings
# ===============================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

# ===============================
# 📊 Batch plot SHAP scatter plots
# ===============================

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
    # --- Plot SHAP dependence ---
    shap.dependence_plot(
        feat,
        shap_values_Explanation.values,
        X,
        interaction_index=None,
        show=False,
        dot_size=24
    )

    # ===== Add black border to dots =====
    ax = plt.gca()
    for coll in ax.collections:
        if hasattr(coll, "set_edgecolor"):
            coll.set_edgecolor('black')
            coll.set_linewidth(0.7)

    # ===== Aesthetic adjustments =====
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)

    # Add SHAP=0 horizontal line
    plt.axhline(y=0, color='black', linestyle='-.', linewidth=2)

    # Clean filename
    safe_feat = feat.replace(' ', '_').replace('/', '_')

    # Save as PDF
    plt.savefig(f'SHAP_Y2_PsychFocus-{i}({safe_feat}).pdf', dpi=1000, format='pdf', bbox_inches='tight')

    plt.show()
