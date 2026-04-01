import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

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
# Boruta feature selection
# ===============================
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

rf = RandomForestClassifier(
    n_jobs=-1,
    class_weight='balanced',
    max_depth=5,
    random_state=42
)

boruta_selector = BorutaPy(
    rf,
    n_estimators='auto',
    verbose=2,
    random_state=42
)

boruta_selector.fit(X_train.values, y_train.values)

selected_features = X_train.columns[boruta_selector.support_].to_list()
rejected_features = X_train.columns[~boruta_selector.support_].to_list()
tentative_features = X_train.columns[boruta_selector.support_weak_].to_list()

print("Selected features:", selected_features)
print("Rejected features:", rejected_features)
print("Tentative features:", tentative_features)

# ===============================
# Stability analysis (20 runs)
# ===============================
ranking_df = pd.DataFrame(
    index=range(1, 21),
    columns=X_train.columns
)

for i in range(20):
    print(f"Iteration {i + 1}")

    boruta_selector = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=0,
        random_state=i,
        max_iter=50
    )

    boruta_selector.fit(X_train.values, y_train.values)

    ranking_df.loc[i + 1] = boruta_selector.ranking_

# ===============================
# Visualization of ranking stability
# ===============================
import seaborn as sns

numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')

median_values = numeric_ranking_df.median()
sorted_columns = median_values.sort_values().index

selected_features = X_train.columns[boruta_selector.support_].to_list()
rejected_features = X_train.columns[~boruta_selector.support_].to_list()
tentative_features = X_train.columns[boruta_selector.support_weak_].to_list()

color_map = {feature: "#02BBC1" for feature in selected_features}
color_map.update({feature: "#E53935" for feature in rejected_features})
color_map.update({feature: "#FFC107" for feature in tentative_features})

plt.figure(figsize=(15, 8))
sns.set(style="whitegrid")

ax = sns.boxplot(
    data=numeric_ranking_df[sorted_columns],
    palette=color_map
)

plt.xticks(rotation=90)

plt.title(
    "Boruta Feature Ranking Distribution",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Features", fontsize=16)
plt.ylabel("Importance ranking", fontsize=16)

for tick, label in zip(range(len(sorted_columns)), ax.get_xticklabels()):
    feature = sorted_columns[tick]
    label.set_color(color_map.get(feature, "black"))

for label in ax.get_xticklabels():
    label.set_fontsize(14)

ax.tick_params(axis='both', which='major', labelsize=14)

handles = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#02BBC1", markersize=12, label='Selected'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#E53935", markersize=12, label='Rejected'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#FFC107", markersize=12, label='Tentative')
]

plt.legend(
    handles=handles,
    title="Feature status",
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "boruta_feature_ranking_distribution.pdf",
    format='pdf',
    bbox_inches='tight',
    dpi=1200
)

plt.show()
