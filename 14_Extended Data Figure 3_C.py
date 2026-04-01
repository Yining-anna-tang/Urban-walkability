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
# XGBoost hyperparameter tuning
# ===============================
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

model_xgb = XGBClassifier(eval_metric='logloss', random_state=8)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)

grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=kfold,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
xgboost = grid_search.best_estimator_

# ===============================
# Predictions
# ===============================
y_pred = xgboost.predict(X_test)
print(classification_report(y_test, y_pred))

# ===============================
# Confusion matrix (default threshold)
# ===============================
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

labels = ['Class 0', 'Class 1']

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Reds",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_default.pdf", dpi=1200)
plt.show()

# ===============================
# Threshold optimization (TSS)
# ===============================
probabilities = xgboost.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 101)

tss_values = []

for threshold in thresholds:
    pred = (probabilities > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    tss = sensitivity + specificity - 1
    tss_values.append(tss)

tss_values = np.array(tss_values)

optimal_threshold = thresholds[np.argmax(tss_values)]
max_tss = tss_values.max()

# ===============================
# Plot TSS curve
# ===============================
plt.figure(figsize=(8, 6), dpi=1200)
plt.plot(thresholds, tss_values)
plt.axvline(optimal_threshold, linestyle='--')
plt.axhline(max_tss, linestyle='--')

plt.xlabel("Probability threshold")
plt.ylabel("TSS")
plt.tight_layout()
plt.savefig("tss_curve.pdf")
plt.show()

# ===============================
# Confusion matrix (optimal threshold)
# ===============================
y_pred_opt = (probabilities > optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_opt))

cm = confusion_matrix(y_test, y_pred_opt)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_optimal.pdf", dpi=1200)
plt.show()

# ===============================
# SHAP decision plot
# ===============================
import shap

explainer = shap.TreeExplainer(xgboost)
shap_values = explainer.shap_values(X_test)

base_value = explainer.expected_value
sample_index = 1

plt.figure(figsize=(10, 5), dpi=1200)
shap.decision_plot(
    base_value,
    shap_values[sample_index],
    X_test.iloc[sample_index],
    show=False,
    link='logit'
)

plt.tight_layout()
plt.savefig("shap_decision_plot.pdf")
