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
# Global plotting configuration
# ===============================
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

# ===============================
# 1) Load dataset and define X, y
# ===============================
data = pd.read_csv("0_dataset.csv", encoding="utf-8")
df = pd.DataFrame(data)

X = df.drop(['Y'], axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 2) Output directory
# ===============================
out_dir = "learning_rate_tuning_outputs"
os.makedirs(out_dir, exist_ok=True)
print(f"Output directory: {out_dir}")

ts_all = datetime.now().strftime("%Y%m%d_%H%M%S")

# ===============================
# 3) Learning rate loop
# ===============================
learning_rates = [round(x / 100, 2) for x in range(1, 11)]
metrics_rows = []

global_best_rmse = np.inf
global_best_lr = None

for lr in learning_rates:
    print("\n" + "="*60)
    print(f"Training model with learning rate = {lr:.2f}")
    print("="*60)

    params_cat = {
        'learning_rate': lr,
        'iterations': 1000,
        'depth': 6,
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 0
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = np.inf
    best_model = None
    fold_scores = []

    # ----- Cross-validation -----
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = CatBoostRegressor(**params_cat)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)

        y_val_pred = model.predict(X_val)
        score = root_mean_squared_error(y_val, y_val_pred)
        fold_scores.append(score)

        print(f"Fold {fold + 1} RMSE: {score:.6f}")

        if score < best_score:
            best_score = score
            best_model = model

    print(f"Best CV RMSE: {best_score:.6f}")
    print(f"Mean CV RMSE: {np.mean(fold_scores):.6f}")

    # ===============================
    # Test set evaluation
    # ===============================
    y_pred = best_model.predict(X_test)

    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    cc = np.corrcoef(y_test, y_pred)[0, 1]

    mean_pred = float(np.mean(y_pred))
    std_pred = float(np.std(y_pred))
    mean_pred_safe = mean_pred if abs(mean_pred) > 1e-12 else 1e-12
    rsd = std_pred / mean_pred_safe

    print("--- Test Metrics ---")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R2  : {r2:.6f}")
    print(f"CC  : {cc:.6f}")

    if rmse < global_best_rmse:
        global_best_rmse = rmse
        global_best_lr = lr

    metrics_rows.append({
        "learning_rate": lr,
        "CV_RMSE_mean": round(np.mean(fold_scores), 6),
        "CV_RMSE_std": round(np.std(fold_scores), 6),
        "Test_RMSE": round(rmse, 6),
        "Test_MAE": round(mae, 6),
        "Test_R2": round(r2, 6),
        "Correlation": round(cc, 6),
        "RSD": round(rsd, 6)
    })

    # ===============================
    # SHAP analysis
    # ===============================
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    lr_str = str(lr).replace('.', 'p')

    shap_path = os.path.join(out_dir, f"shap_values_lr_{lr_str}.csv")
    shap_df.to_csv(shap_path, index=False)

    feature_importance = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=ranking.head(20),
        palette="viridis"
    )

    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"feature_importance_lr_{lr_str}.pdf")
    plt.savefig(fig_path, dpi=1200, bbox_inches='tight')
    plt.close()

    # ===============================
    # Save best model
    # ===============================
    model_path = os.path.join(out_dir, f"catboost_lr_{lr_str}.cbm")
    best_model.save_model(model_path)

# ===============================
# 4) Save metrics to Excel
# ===============================
metrics_df = pd.DataFrame(metrics_rows)

excel_path = os.path.join(
    out_dir,
    f"learning_rate_metrics_{ts_all}.xlsx"
)

with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    metrics_df.to_excel(writer, index=False, sheet_name="metrics")

# ===============================
# 5) Plot learning rate curve
# ===============================
plt.figure(figsize=(8, 6))
plt.plot(metrics_df["learning_rate"], metrics_df["Test_RMSE"], marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Test RMSE")
plt.grid(True)

curve_path = os.path.join(out_dir, "learning_rate_vs_rmse.pdf")
plt.savefig(curve_path, dpi=1200, bbox_inches='tight')
plt.close()

print("\nBest learning rate:", global_best_lr)
print("Best test RMSE:", global_best_rmse)
print("All outputs saved to:", out_dir)
