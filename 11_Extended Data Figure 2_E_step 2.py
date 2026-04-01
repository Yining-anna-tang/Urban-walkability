import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")

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
# Train XGBoost classifier
# ===============================
import xgboost as xgb
from sklearn.metrics import accuracy_score

model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# SHAP explanation
# ===============================
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)

# ===============================
# Define feature groups
# ===============================
continuous_features = ['LSC', 'AGE', 'PAI', 'EDU']
categorical_features = ['EBD', 'LSI', 'HSAA', 'EI', 'DMF', 'EPK']

continuous_df = shap_values_df[continuous_features]
categorical_df = shap_values_df[categorical_features]

features = continuous_df.columns.tolist() + categorical_df.columns.tolist()

# ===============================
# SHAP scatter + LOWESS plots
# ===============================
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for i in range(len(axes)):
    if i < len(features):
        feature = features[i]

        if feature in X_test.columns:
            ax = axes[i]

            scatter = ax.scatter(
                X_test[feature],
                shap_values_df[feature],
                s=30,
                c=X_test[feature],
                cmap='coolwarm',
                edgecolor='k'
            )

            ax.axhline(y=0, color='red', linestyle='-.', linewidth=1)

            lowess_fit = lowess(
                shap_values_df[feature],
                X_test[feature],
                frac=0.3
            )

            ax.plot(
                lowess_fit[:, 0],
                lowess_fit[:, 1],
                color='#B5B5B5',
                linewidth=2
            )

            ax.set_xlabel(feature, fontsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Feature value")

        else:
            axes[i].axis('off')
    else:
        axes[i].axis('off')

plt.subplots_adjust(hspace=0.4, wspace=0.8)

plt.savefig(
    "shap_scatter_lowess_10panels.pdf",
    format='pdf',
    bbox_inches='tight',
    dpi=1200
)

plt.tight_layout()
plt.show()
