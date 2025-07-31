# ------------------------------Import the required libraries------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import KBinsDiscretizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ------------------------------Loading the training and testing data------------------------------
excel_path = "LTF Challenge data with dictionary.xlsx"
train_df = pd.read_excel(excel_path, sheet_name="TrainData")
test_df = pd.read_excel(excel_path, sheet_name="TestData")

# Drop identifier
train_df_dropped = train_df.drop(columns=["FarmerID"])
test_df_dropped = test_df.drop(columns=["FarmerID"])

# ------------------------------Null Value Imputation--------------------------------
num_cols = train_df_dropped.select_dtypes(include=np.number).columns.tolist()
cat_cols = train_df_dropped.select_dtypes(exclude=np.number).columns.tolist()
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
train_df_dropped[num_cols] = num_imputer.fit_transform(train_df_dropped[num_cols])
train_df_dropped[cat_cols] = cat_imputer.fit_transform(train_df_dropped[cat_cols])
test_df_dropped[num_cols] = num_imputer.transform(test_df_dropped[num_cols])
test_df_dropped[cat_cols] = cat_imputer.transform(test_df_dropped[cat_cols])

# ------------------------------Splitting the min/max columns------------------------------
# List of columns to process
temp_cols = [
    "K022-Ambient temperature (min & max)",
    "R020-Ambient temperature (min & max)",
    "R021-Ambient temperature (min & max)",
    "R022-Ambient temperature (min & max)",
    "K021-Ambient temperature (min & max)"
]
# Loop through each column and split values
for col in temp_cols:
    base_name = col.split(" ")[0]  # e.g., 'K022'
    train_df_dropped[[f"{base_name}_Temp_Min", f"{base_name}_Temp_Max"]] = (
        train_df_dropped[col]
        .str.split("/", expand=True)
        .astype(float)
    )
    test_df_dropped[[f"{base_name}_Temp_Min", f"{base_name}_Temp_Max"]] = (
        test_df_dropped[col]
        .str.split("/", expand=True)
        .astype(float)
    )
    train_df_dropped.drop(columns=[col], inplace=True)
    test_df_dropped.drop(columns=[col], inplace=True)

# ------------------------------Feature Engineering--------------------------------
# Step 1: Ordinal encoding of categorical variables
X_temp = train_df_dropped.drop(columns=["Target_Variable/Total Income"])
X_test_temp = test_df_dropped.drop(columns=["Target_Variable/Total Income"])

cat_cols = X_temp.select_dtypes(exclude='number').columns.tolist()
num_cols = X_temp.select_dtypes(include='number').columns.tolist()

# Fit OrdinalEncoder on train and apply on both train and test
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_temp[cat_cols] = ordinal_encoder.fit_transform(X_temp[cat_cols])
X_test_temp[cat_cols] = ordinal_encoder.transform(X_test_temp[cat_cols])

# Step 3: Standard scale all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_temp)
X_test_scaled = scaler.transform(X_test_temp)

# Step 4: Apply PCA to retain 98% variance
pca = PCA(n_components=0.98, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ------------------------------Evaluating which model works best on a stratified sample--------------------------------

# Step 1: Stratified sampling using binned y values
y = train_df["Target_Variable/Total Income"]
bins = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='quantile')
y_bins = bins.fit_transform(y.values.reshape(-1, 1)).flatten()
_, X_sampled, _, y_sampled = train_test_split(X_pca, y, stratify=y_bins, test_size=10000, random_state=42)

# Step 2: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

# Step 3: Define models
models = {
    "XGBRegressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "CatBoostRegressor": CatBoostRegressor(verbose=0, random_state=42)
}

# Step 4: Fit and evaluate MAPE
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)

# Since each model has similar performance, we build an ensemble taking simple average of all the four models

# ------------------------------Building the base ensemble and training it------------------------------

# Train-test 80-20 split
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=43)
# Step 2: Initialize models
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=43)
rf_model = RandomForestRegressor(n_estimators=100, random_state=43)
svr_model = SVR()
cb_model = CatBoostRegressor(verbose=0, random_state=43)

# Step 3: Train each model
xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)
cb_model.fit(X_train, y_train)

# Step 4: Predict on validation set
pred_xgb = xgb_model.predict(X_val)
pred_rf = rf_model.predict(X_val)
pred_svr = svr_model.predict(X_val)
pred_cb = cb_model.predict(X_val)

# Step 5: Simple average ensemble
final_pred = (pred_xgb + pred_rf + pred_svr + pred_cb) / 4

# Step 6: Calculate MAPE
final_mape = mean_absolute_percentage_error(y_val, final_pred)
print(f"Simple Average Ensemble MAPE on Validation Set: {final_mape:.4f}")

# ------------------------------Hyperparameter Tuning using stratified sample--------------------------------
# Step 1: Stratified sampling for 10,000 points
bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
y_bins = bins.fit_transform(y.values.reshape(-1, 1)).flatten()
_, X_sample, _, y_sample = train_test_split(X_pca, y, stratify=y_bins, test_size=7000, random_state=42)

# Step 2: Split sample into training and validation sets
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Step 3: Define hyperparameter search spaces
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0]
}

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

svr_param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2],
    'kernel': ['rbf', 'linear']
}

cb_param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 300]
}

# Step 4: Tune each model using RandomizedSearchCV
scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_param_grid, scoring=scorer, cv=3, n_iter=10, random_state=42)
xgb_search.fit(X_train_s, y_train_s)

rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, scoring=scorer, cv=3, n_iter=10, random_state=42)
rf_search.fit(X_train_s, y_train_s)

svr_search = RandomizedSearchCV(SVR(), svr_param_grid, scoring=scorer, cv=3, n_iter=10, random_state=42)
svr_search.fit(X_train_s, y_train_s)

cb_search = RandomizedSearchCV(CatBoostRegressor(verbose=0, random_state=42), cb_param_grid, scoring=scorer, cv=3, n_iter=10, random_state=42)
cb_search.fit(X_train_s, y_train_s)

# Step 5: Retrieve best estimators
best_xgb = xgb_search.best_estimator_
best_rf = rf_search.best_estimator_
best_svr = svr_search.best_estimator_
best_cb = cb_search.best_estimator_

#------------------------------Training the ensemble on the optimal hyperparameter values-------------------------------- 

# Step 6: Train ensemble on full training data with best hyperparameters
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(X_pca, y, test_size=0.2, random_state=42)

best_xgb.fit(X_train_full, y_train_full)
best_rf.fit(X_train_full, y_train_full)
best_svr.fit(X_train_full, y_train_full)
best_cb.fit(X_train_full, y_train_full)

# Step 7: Predict and compute MAPE
pred_xgb = best_xgb.predict(X_val_full)
pred_rf = best_rf.predict(X_val_full)
pred_svr = best_svr.predict(X_val_full)
pred_cb = best_cb.predict(X_val_full)

final_pred = (pred_xgb + pred_rf + pred_svr + pred_cb) / 4
final_mape = mean_absolute_percentage_error(y_val_full, final_pred)
print(f"Final Ensemble MAPE (with tuned hyperparameters): {final_mape:.4f}")
print(f"MAPE for XGBRegressor: {mean_absolute_percentage_error(y_val_full, pred_xgb):.4f}")
print(f"MAPE for RandomForestRegressor: {mean_absolute_percentage_error(y_val_full, pred_rf):.4f}")
print(f"MAPE for SVR: {mean_absolute_percentage_error(y_val_full, pred_svr):.4f}")
print(f"MAPE for CatBoostRegressor: {mean_absolute_percentage_error(y_val_full, pred_cb):.4f}")

# ------------------------------Final Predictions on Test Data--------------------------------
# Step 4: Run through tuned ensemble models
pred_xgb_test = best_xgb.predict(X_test_pca)
pred_rf_test = best_rf.predict(X_test_pca)
pred_svr_test = best_svr.predict(X_test_pca)
pred_cb_test = best_cb.predict(X_test_pca)

# Step 5: Simple average ensemble prediction
final_test_pred = (pred_xgb_test + pred_rf_test + pred_svr_test + pred_cb_test) / 4
# Step 7: Create output DataFrame and export
output_df = pd.DataFrame({
    "FarmerID": test_df["FarmerID"].astype(str),
    "Target_Variable/Total Income": final_test_pred
})
output_df.to_excel("Ensemble_Predicted_Incomes.xlsx", index=False)



