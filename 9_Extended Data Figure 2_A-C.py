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
df = pd.read_csv("0_dataset.csv", encoding="utf-8")

from sklearn.model_selection import train_test_split

X = df.drop(['Y'], axis=1)
y = df['Y']

print("Feature matrix shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# RFE feature ranking
# ===============================
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

selection_results = pd.DataFrame(
    columns=['Feature', 'Importance', 'MSE', 'R2']
)

rf_reg = RandomForestRegressor(random_state=42)

rfe = RFE(estimator=rf_reg, n_features_to_select=1, step=1)
rfe.fit(X_train, y_train)

feature_ranking = rfe.ranking_

rfe_features = pd.DataFrame({
    'Feature': X_train.columns,
    'Ranking': feature_ranking
}).sort_values(by='Ranking')

# ===============================
# Recursive feature evaluation
# ===============================
selected_features = []

for i in range(len(rfe_features)):
    current_feature = rfe_features.iloc[i]['Feature']
    selected_features.append(current_feature)

    X_train_subset = X_train[selected_features]
    X_test_subset = X_test[selected_features]

    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train_subset, y_train)

    importance = rf_reg.feature_importances_[len(selected_features) - 1]

    y_pred = rf_reg.predict(X_test_subset)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    selection_results.loc[len(selection_results)] = [
        current_feature,
        importance,
        mse,
        r2
    ]

selection_results = selection_results.iloc[0:30]

# ===============================
# Plot R2 progression
# ===============================
n_features = 12

fig, ax1 = plt.subplots(figsize=(16, 6))

norm = plt.Normalize(
    selection_results['Importance'].min(),
    selection_results['Importance'].max()
)

colors = plt.cm.Blues(norm(selection_results['Importance']))

ax1.bar(
    selection_results['Feature'],
    selection_results['Importance'],
    color=colors
)

ax1.set_ylabel("Feature Importance")

ax2 = ax1.twinx()

ax2.plot(
    selection_results['Feature'],
    selection_results['R2'],
    color="red",
    marker='o'
)

ax2.set_ylabel("R2 Score")

plt.xticks(rotation=75)

plt.tight_layout()
plt.savefig("rfe_r2_progression.pdf", dpi=1200)
plt.show()

# ===============================
# Plot MSE progression
# ===============================
fig, ax1 = plt.subplots(figsize=(16, 6))

ax1.bar(
    selection_results['Feature'],
    selection_results['Importance'],
    color=colors
)

ax2 = ax1.twinx()

ax2.plot(
    selection_results['Feature'],
    selection_results['MSE'],
    color="black",
    marker='o'
)

ax2.set_ylabel("Mean Squared Error")

plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("rfe_mse_progression.pdf", dpi=1200)
plt.show()

# ===============================
# Select optimal feature subset
# ===============================
optimal_features = selection_results.iloc[0:11]['Feature']

X = df[optimal_features]
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Hyperparameter tuning
# ===============================
from sklearn.model_selection import GridSearchCV, KFold

rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=KFold(n_splits=5),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_

# ===============================
# Model evaluation
# ===============================
from sklearn.metrics import mean_absolute_error

y_pred_train = best_rf_model.predict(X_train)
y_pred_test = best_rf_model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Train RMSE:", rmse_train)
print("Test RMSE:", rmse_test)
print("Train R2:", r2_train)
print("Test R2:", r2_test)

# ===============================
# Prediction scatter plot
# ===============================
residuals = y_test - y_pred_test

fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(
    y_test,
    y_pred_test,
    c=residuals,
    cmap='jet',
    s=30
)

ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.colorbar(scatter)
plt.xlabel("Observed")
plt.ylabel("Predicted")

plt.tight_layout()
plt.savefig("prediction_scatter.pdf", dpi=1200)
plt.show()

# ===============================
# SHAP analysis
# ===============================
import shap

explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
shap_df.to_csv("shap_values.csv", index=False)

# ===============================
# GAM interpretation for one feature
# ===============================
from pygam import LinearGAM, s

feature_name = X_test.columns[0]

X_gam = X_test[feature_name].values.reshape(-1, 1)
y_gam = shap_df[feature_name].values

gam = LinearGAM(s(0)).fit(X_gam, y_gam)

XX = gam.generate_X_grid(term=0)
y_pred_gam = gam.predict(XX)

plt.figure(figsize=(6, 4))
plt.plot(XX, y_pred_gam)
plt.xlabel(feature_name)
plt.ylabel("SHAP value")

plt.tight_layout()
plt.savefig("gam_shap_curve.pdf", dpi=1200)
plt.show()
