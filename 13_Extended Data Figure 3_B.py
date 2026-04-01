import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ===============================
# Load dataset
# ===============================
df = pd.read_csv("0_dataset_binary.csv", encoding="utf-8")

# ===============================
# Train-test split
# ===============================
from sklearn.model_selection import train_test_split

X = df.drop(['Y', 'Y_binary'], axis=1)
y = df['Y_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ===============================
# LightGBM feature importance
# ===============================
import lightgbm as lgb

lgbm_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgbm_clf.fit(X_train, y_train)

feature_importances = lgbm_clf.feature_importances_

lgbm_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# ===============================
# Incremental feature evaluation
# ===============================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

top_features = lgbm_feature_importance.sort_values(by='Importance', ascending=False)

selection_A_LGBM = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])
selection_A_LogReg = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])
selection_A_RF = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])
selection_A_XGB = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])

selected_features = []

kf = KFold(n_splits=4, shuffle=True, random_state=42)
n_splits = kf.get_n_splits()
fold_columns = [f'Fold_{i+1}_ROC' for i in range(n_splits)]

for i in range(len(top_features)):
    current_feature = top_features.iloc[i]['Feature']
    selected_features.append(current_feature)

    fold_roc_scores_LGBM = []
    fold_roc_scores_LogReg = []
    fold_roc_scores_RF = []
    fold_roc_scores_XGB = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold = X_train.iloc[train_idx][selected_features]
        X_val_fold = X_train.iloc[val_idx][selected_features]

        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]

        # LightGBM
        lgbm_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
        lgbm_clf.fit(X_train_fold, y_train_fold)
        y_val_proba = lgbm_clf.predict_proba(X_val_fold)[:, 1]
        fold_roc_scores_LGBM.append(roc_auc_score(y_val_fold, y_val_proba))

        # Logistic Regression
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        logreg_clf = LogisticRegression(max_iter=200, random_state=42)
        logreg_clf.fit(X_train_fold_scaled, y_train_fold)
        y_val_proba = logreg_clf.predict_proba(X_val_fold_scaled)[:, 1]
        fold_roc_scores_LogReg.append(roc_auc_score(y_val_fold, y_val_proba))

        # Random Forest
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(X_train_fold, y_train_fold)
        y_val_proba = rf_clf.predict_proba(X_val_fold)[:, 1]
        fold_roc_scores_RF.append(roc_auc_score(y_val_fold, y_val_proba))

        # XGBoost
        xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_clf.fit(X_train_fold, y_train_fold)
        y_val_proba = xgb_clf.predict_proba(X_val_fold)[:, 1]
        fold_roc_scores_XGB.append(roc_auc_score(y_val_fold, y_val_proba))

    mean_roc_score_LGBM = np.mean(fold_roc_scores_LGBM)
    mean_roc_score_LogReg = np.mean(fold_roc_scores_LogReg)
    mean_roc_score_RF = np.mean(fold_roc_scores_RF)
    mean_roc_score_XGB = np.mean(fold_roc_scores_XGB)

    def append_result(df_res, scores, mean_score):
        row = {
            'Feature': current_feature,
            'Importance': top_features.iloc[i]['Importance'],
            'Mean_ROC': mean_score
        }
        for j, score in enumerate(scores):
            row[fold_columns[j]] = score
        return pd.concat([df_res, pd.DataFrame([row])], ignore_index=True)

    selection_A_LGBM = append_result(selection_A_LGBM, fold_roc_scores_LGBM, mean_roc_score_LGBM)
    selection_A_LogReg = append_result(selection_A_LogReg, fold_roc_scores_LogReg, mean_roc_score_LogReg)
    selection_A_RF = append_result(selection_A_RF, fold_roc_scores_RF, mean_roc_score_RF)
    selection_A_XGB = append_result(selection_A_XGB, fold_roc_scores_XGB, mean_roc_score_XGB)

# ===============================
# Plot AUC vs feature count
# ===============================
from matplotlib.lines import Line2D

def plot_auc_per_feature(selection_A, label, color, ax):
    feature_count = np.arange(1, len(selection_A) + 1)
    std_error = selection_A.iloc[:, 3:].std(axis=1) / np.sqrt(selection_A.iloc[:, 3:].shape[1])
    yerr = 1.96 * std_error

    ax.errorbar(
        feature_count,
        selection_A['Mean_ROC'],
        yerr=yerr,
        label=label,
        color=color,
        capsize=5,
        alpha=0.8
    )

    ax.set_xlabel('Number of features')
    ax.set_ylabel('AUC')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

fig, ax = plt.subplots(figsize=(10, 6))

plot_auc_per_feature(selection_A_LGBM, 'LightGBM', 'orange', ax)
plot_auc_per_feature(selection_A_LogReg, 'Logistic Regression', 'green', ax)
plot_auc_per_feature(selection_A_RF, 'Random Forest', 'red', ax)
plot_auc_per_feature(selection_A_XGB, 'XGBoost', 'blue', ax)

legend_lines = [
    Line2D([0], [0], color='orange', lw=2),
    Line2D([0], [0], color='green', lw=2),
    Line2D([0], [0], color='red', lw=2),
    Line2D([0], [0], color='blue', lw=2)
]

ax.legend(
    handles=legend_lines,
    labels=['LightGBM', 'Logistic Regression', 'Random Forest', 'XGBoost'],
    loc='lower right'
)

plt.tight_layout()
plt.savefig(
    "optimal_feature_count_auc.pdf",
    format='pdf',
    bbox_inches='tight',
    dpi=1200
)
plt.show()
