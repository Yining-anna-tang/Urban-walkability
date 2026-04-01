# 20.7 Learning Curve with Light Gray Confidence Bands (Y2 = Psychological Focus)
# Transparent background, no grid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.ensemble import RandomForestRegressor

# ======================
# Configuration
# ======================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# ======================
# Load dataset
# ======================
df = pd.read_csv('0_dataset.csv')

X = df.drop(['Y'], axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ======================
# Model and learning curve
# ======================
model = RandomForestRegressor(random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train_sizes, train_scores, valid_scores = learning_curve(
    model,
    X_train,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 100),
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Convert negative MSE to positive
train_scores_mse = -train_scores
valid_scores_mse = -valid_scores

train_mean = np.mean(train_scores_mse, axis=1)
train_std = np.std(train_scores_mse, axis=1)
valid_mean = np.mean(valid_scores_mse, axis=1)
valid_std = np.std(valid_scores_mse, axis=1)

# ======================
# Save path
# ======================
save_dir = "results_learning_curve_Y2"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(
    save_dir,
    "20_7_learning_curve_Y2_psychological_focus.pdf"
)

# ======================
# Plotting (transparent background, no grid)
# ======================
fig, ax = plt.subplots(figsize=(10, 8), dpi=1200)

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")
ax.grid(False)

# Plot curves
ax.plot(train_sizes, train_mean, linestyle='--', color='black', label='Training Error')
ax.plot(train_sizes, valid_mean, linestyle='-', color='black', label='Validation Error')

# Confidence intervals
ax.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2,
    color='darkgray'
)

ax.fill_between(
    train_sizes,
    valid_mean - valid_std,
    valid_mean + valid_std,
    alpha=0.2,
    color='gray'
)

# Axis and legend
ax.set_title('', fontsize=14, fontweight='bold')
ax.set_xlabel('', fontsize=12, fontweight='bold')
ax.set_ylabel('', fontsize=12, fontweight='bold')

ax.tick_params(axis='both', labelsize=30)
ax.set_xlim(left=0)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=2,
    fontsize=24,
    frameon=False
)

plt.subplots_adjust(top=0.8)
plt.tight_layout()

# Save figure
plt.savefig(
    save_path,
    format='pdf',
    bbox_inches='tight',
    transparent=True
)

plt.show()

print(f"Figure saved to: {save_path}")
