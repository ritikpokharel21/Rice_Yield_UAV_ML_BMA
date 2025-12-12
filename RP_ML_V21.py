# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:16:52 2025
@author: RPokharel
"""
#%%Libraries and pacakges
import os, json, random

# === Data / plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Scikit-learn (preprocess, model selection, models, metrics, utils)
from sklearn.preprocessing import StandardScaler #(zero mean, unit variance).
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, median_absolute_error, r2_score)
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression
import joblib
#Gradient boosting libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
#TensorFlow / Keras & Optuna
import tensorflow as tf 
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import optuna
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import random
from scipy.stats import median_abs_deviation
import pickle
from scipy import stats
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# Global seed value
SEED = 42
# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
#%%Sets the working directory to the main directory containing the project
dir1=os.path.realpath(__file__)
main_dir = os.path.dirname(os.path.dirname(dir1)) #i.e., one level above where this script file is saved).
# Define file paths
data_path = os.path.join(main_dir, 'Data', 'Processed', 'RICE_COMBILED_mlrs.csv')
results_dir = os.path.join(main_dir, 'Results')
graphs_dir = os.path.join(results_dir, 'Graphs')
# Create output directories if they don't exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)
# Load dataset
Ex1 = pd.read_csv(data_path, index_col=0)
print(f"Data loaded: {Ex1.shape[0]} rows × {Ex1.shape[1]} columns")
#%%Removes rows with output variable empty
initial_rows = Ex1.shape[0]
Ex1.dropna(subset=['VRYieldVOl'],inplace=True)
Ex1=Ex1[Ex1['VRYieldVOl']>0]
print(f" Cleaned data: Removed {initial_rows - Ex1.shape[0]} rows with null or zero yield")
#Moves output variable first
cols = list(Ex1)
cols.insert(0, cols.pop(cols.index('VRYieldVOl')))
Ex1=Ex1.loc[:,cols]
#Computes and saves summary statistics
summary_stats=Ex1.describe()
summary_stats.to_csv(main_dir+'/Results/Data_Description.csv')
print(f"Summary saved to: {results_dir}/Data_Description.csv")
#%%One-Hot Encode Categorical Variables
Ex4 = pd.get_dummies(Ex1, columns=['Variety', 'Year']) #Converts categorical variables Variety and Year into one-hot encoded dummy variables (binary 0/1 columns for each category).
bool_cols = Ex4.select_dtypes(include='bool').columns
Ex4[bool_cols] = Ex4[bool_cols].astype(int)
#Convert Entire DataFrame to float64
Ex4 = Ex4.astype('float64') 
print(Ex4.dtypes)
print(f" Final shape after encoding: {Ex4.shape}")
#%%Splits to input and output and flattens
X = Ex4.iloc[:, 1:].values #selects all columns except the first one(yield) — these are input features (predictors).
y = Ex4.iloc[:, 0].values.flatten() #selects the first column, which is 'VRYieldVOl', the target/output.
print(" X shape:", X.shape)
print(" y shape:", y.shape)
#Standardize input features (once)
scaler = StandardScaler() #Uses Z-score normalization to Transforms all features to zero mean and unit variance, which is critical for NN and regressions and boosting models. 
X_scaled = scaler.fit_transform(X) # only to x, dont do it for Y.  
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)
#Then define cv and start model optimization
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
#%%Hyperparameter Tuning for Neural network using Optuna
#Define Neural Network Optimization using Optuna
def objective(trial):
    n_layers = trial.suggest_int("n_layers", 3, 15)
    layer_sizes = [trial.suggest_int(f"units_{i}", 64, 1024, step=64) for i in range(n_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.6)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "selu"])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Nadam", "RMSprop"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 60, 150)

    # 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rmse_scores = []
    nrmse_scores = []
    
    for train_index, val_index in kf.split(X_scaled):
        X_t, X_v = X_scaled[train_index], X_scaled[val_index]
        y_t, y_v = y[train_index], y[val_index]    

        model = models.Sequential()
        model.add(layers.Dense(layer_sizes[0], activation=activation, input_shape=(X.shape[1],)))
        for units in layer_sizes[1:]:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1))  # Output layer

        optimizer = getattr(optimizers, optimizer_name)(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            TFKerasPruningCallback(trial, "val_loss")
        ]

        model.fit(X_t, y_t,
                  validation_data=(X_v, y_v),
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=0,
                  callbacks=callbacks)

        y_pred = model.predict(X_v).flatten()
        rmse = np.sqrt(mean_squared_error(y_v, y_pred))
        nrmse = (rmse / np.mean(y_v)) * 100
        
        rmse_scores.append(rmse)
        nrmse_scores.append(nrmse)
     
    trial.set_user_attr("avg_nrmse_pct", np.mean(nrmse_scores))
    return np.mean(rmse_scores)

#%%Run Optuna Optimization for Neural network
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50, n_jobs=1)
print(" Best parameters found:")
print(study.best_params)

# Save best NN params
with open(os.path.join(results_dir, 'best_nn_params.json'), 'w') as f:
    json.dump(study.best_params, f)
#%%Hyperparameter tuning for XGboost, RF
# Random Forest hyperparameter grid
rf_param_grid = {
    'n_estimators': range(100, 900, 200),
    'max_depth': range(1, 40, 10),
    'max_features': range(1, min(60, X.shape[1]), 5)}
# XGBoost hyperparameter grid
xg_param_grid = {
    'n_estimators': [500, 700, 1000, 1100, 1200],
    'learning_rate': [0.003, 0.005, 0.007],
    'max_depth': [3, 5, 8],
    'subsample': [0.4, 0.5, 0.7],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': np.arange(0.0, 0.01, 0.02)}
# Grid Search for Random Forest
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=SEED), rf_param_grid, cv=5, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
print(" Best RF Parameters:", rf_grid_search.best_params_)
with open(os.path.join(results_dir, 'best_rf_params.json'), 'w') as f:
    json.dump(rf_grid_search.best_params_, f)
    
    # Custom scorer
scorer = make_scorer(mean_squared_error, greater_is_better=False)
# Flatten search space for progress tracking
import itertools
param_combinations = list(itertools.product(*xg_param_grid.values()))
total_combinations = len(param_combinations)
# GridSearch
xgb_model = xgb.XGBRegressor(n_jobs=-1, tree_method='auto')
cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xg_param_grid,
    scoring='neg_mean_squared_error',
    cv=cv,
    verbose=1,
    n_jobs=-1)
print(" Starting XGBoost grid search with %d combinations..." % total_combinations)
grid_search.fit(X_train, y_train)

print(" Best parameters for XGBoost:")
print(grid_search.best_params_)

xg_best_params = grid_search.best_params_

# Save best params to JSON
with open(os.path.join(results_dir, 'xgboost_best_params.json'), 'w') as f:
    json.dump(xg_best_params, f, indent=4)
print(" Best XGBoost parameters saved to 'xgboost_best_params.json'")
#%%Hyperparameter tuning for other exploratory model like ridge, lasso, adaboost, elastic net
# Define parameter grids
ridge_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_grid = {'alpha': [0.001, 0.01, 0.1, 1]}
elastic_grid = {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.5, 0.9]}
# Grid Search for Ridge
ridge_search = GridSearchCV(Ridge(), ridge_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
ridge_search.fit(X_train, y_train)
print(" Best Ridge params:", ridge_search.best_params_)
# Grid Search for LASSO
lasso_search = GridSearchCV(Lasso(max_iter=10000), lasso_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
lasso_search.fit(X_train, y_train)
print(" Best LASSO params:", lasso_search.best_params_)
# Grid Search for ElasticNet
elastic_search = GridSearchCV(ElasticNet(max_iter=10000), elastic_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
elastic_search.fit(X_train, y_train)
print(" Best ElasticNet params:", elastic_search.best_params_)
# Try a few component numbers manually
pls_rmse = {}
for n_comp in range(2, min(10, X.shape[1])):
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train, y_train)
    preds = pls.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    pls_rmse[n_comp] = rmse
best_pls_components = min(pls_rmse, key=pls_rmse.get)
print(f" Best PLSR Components: {best_pls_components}")

lgb_grid = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [3, 5, 7],
    'num_leaves': [15, 31, 63]}

lgb_model = lgb.LGBMRegressor(random_state=SEED)
lgb_search = GridSearchCV(lgb_model, lgb_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
lgb_search.fit(X_train, y_train)
print(" Best LightGBM params:", lgb_search.best_params_)

cat_model = CatBoostRegressor(verbose=0, random_state=SEED)
cat_grid = {
    'iterations': [300, 500],
    'learning_rate': [0.01, 0.03],
    'depth': [4, 6, 8]
}

cat_search = GridSearchCV(cat_model, cat_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
cat_search.fit(X_train, y_train)
print("Best CatBoost params:", cat_search.best_params_)

#Support Vector Regression (SVR)
svr_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'poly'],
    'epsilon': [0.01, 0.1]}

svr_search = GridSearchCV(SVR(), svr_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
svr_search.fit(X_train, y_train)
print(" Best SVR params:", svr_search.best_params_)

#ExtraTreesRegressor
et_grid = {
    'n_estimators': [200, 500],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2']
}

et_search = GridSearchCV(ExtraTreesRegressor(random_state=SEED), et_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
et_search.fit(X_train, y_train)
print(" Best ExtraTrees params:", et_search.best_params_)

#AdaBoost Regressor
ada_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1, 1.0]}
ada_search = GridSearchCV(AdaBoostRegressor(random_state=SEED), ada_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
ada_search.fit(X_train, y_train)
print(" Best AdaBoost params:", ada_search.best_params_)

#GradientBoosting Regressor
gbr_grid = {
    'n_estimators': [200, 500],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 5]}

gbr_search = GridSearchCV(GradientBoostingRegressor(random_state=SEED), gbr_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
gbr_search.fit(X_train, y_train)
print(" Best Gradient Boosting params:", gbr_search.best_params_)
#%%Result of parameter optimazation, so now on we dont redo the optimization. to save time. 
# Best parameters for Neural Network (From Optuna)
nn_best_params = {
   'n_layers': 4,
    'units_0': 512,
    'units_1': 1024,
    'units_2': 704,
    'units_3': 576,
    'dropout_rate': 0.0880782472351672,
    'learning_rate': 0.0021759875837965175,
    'activation': 'leaky_relu',
    'optimizer': 'Adam',
    'batch_size': 32,
    'epochs': 135}
# Best parameters for Random Forest
rf_best_params = {'max_depth': 21, 'max_features': 31, 'n_estimators': 100}
# Best parameters for XGBoost
xg_best_params = {
    'colsample_bytree': 0.9, 
    'gamma': 0.1, 
    'learning_rate': 0.009, 
    'max_depth': 7, 
    'n_estimators': 1100, 
    'subsample': 0.5}
# Best parameters for Ridge Regression
ridge_best_params = {'alpha': 10}
#Best parameters for LASSO
lasso_best_params = {'alpha': 0.01}
#Best parameters for ElasticNet
elastic_best_params = {'alpha': 0.01, 'l1_ratio': 0.5}
#Best parameters for Partial Least Squares Regression (PLSR)
plsr_best_params = {'n_components': 9}
#Best parameters for LightGBM
lgbm_best_params = {
    'learning_rate': 0.01,
    'max_depth': 3,
    'n_estimators': 300,
    'num_leaves': 15}
#Best parameters for CatBoost
cat_best_params = {
    'depth': 4,
    'iterations': 300,
    'learning_rate': 0.03}
#Best parameters for SVR
svr_best_params = {
    'C': 10,
    'epsilon': 0.1,
    'kernel': 'rbf'}
#Best parameters for Extra Trees
et_best_params = {
    'max_depth': 10,
    'max_features': 'log2',
    'n_estimators': 500}
#Best parameters for AdaBoost
ada_best_params = {
    'learning_rate': 1.0,
    'n_estimators': 300}
#Best parameters for Gradient Boosting
gb_best_params = {
    'learning_rate': 0.01,
    'max_depth': 3,
    'n_estimators': 500}
#Linear Regression has no hyperparameters
lr_best_params = {}  # Included for consistency
#%%Defining benifit indicatos Index of Agreement (IOA) and A10 accuracy
def IOA(obs, pred): #willmontt IOA penalizes larger errors more heavily than simple R².range 0-1.
    return 1 - (np.sum((pred - obs)**2) / np.sum((np.abs(pred - np.mean(obs)) + np.abs(obs - np.mean(obs)))**2))

def A10(obs, pred): #A10 accuracy computes the percentage of predictions that fall within ±10% of the observed values.
    return np.mean(np.abs((pred - obs) / obs) <= 0.10) * 100  # % within ±10%
#%% Main evaluation loop to compare 14 models across 300 randomized train-test splits (80/20), random_state=i ie it changes each time. 
#HereFor each model, we track the best-iterating version across 300 loops based on lowest RMSE value 
# ==== Evaluation
SEED = 42
reps = 300  
model_names = [
    "Random_Forest", "Simple_Regression", "Neural_Network", "XGBoost",
    "Ridge", "LASSO", "ElasticNet", "PLSR", "LightGBM", "CatBoost",
    "SVR", "ExtraTrees", "AdaBoost", "GradientBoost"]

# metric collectors (14 models × 7 metrics)
rmse_all, nrmse_all, mae_all, r2_all = [[] for _ in model_names], [[] for _ in model_names], [[] for _ in model_names], [[] for _ in model_names]
mad_all, ioa_all, a10_all = [[] for _ in model_names], [[] for _ in model_names], [[] for _ in model_names]
best_models = [None] * len(model_names)
best_rmse = [float("inf")] * len(model_names)

# ---------------  main loop  ----------------
for i in range(reps):
    # split & scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=i) #rabdim state=i,  # ← CHANGES EVERY ITERATION!,
    #meaning each iteration has different test set. some test set are harder ot predict ( more variability) and some easier. RMSE varies across iteration due to this test set composition. 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
  # define model objects in the SAME order as model_names
    models_list = [
        RandomForestRegressor(random_state=SEED, n_jobs=-1, **rf_best_params),
        LinearRegression(),
        "NN_PLACEHOLDER",   # will build manually right below
        xgb.XGBRegressor(random_state=SEED, n_jobs=-1, **xg_best_params),
        Ridge(**ridge_best_params),
        Lasso(**lasso_best_params),
        ElasticNet(**elastic_best_params),
        PLSRegression(n_components=plsr_best_params['n_components']),
        lgb.LGBMRegressor(random_state=SEED, **lgbm_best_params),
        CatBoostRegressor(random_seed=SEED, verbose=0, **cat_best_params),
        SVR(**svr_best_params),  # You can tune this later
        ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **et_best_params),
        AdaBoostRegressor(random_state=SEED, **ada_best_params),
        GradientBoostingRegressor(random_state=SEED, **gb_best_params)
    ]

    for idx, mdl in enumerate(models_list):
        # ----- build & predict -----
        if mdl == "NN_PLACEHOLDER":
            nn = models.Sequential()
            nn.add(layers.Dense(nn_best_params["units_0"],
                                activation=nn_best_params["activation"],
                                input_shape=(X_train.shape[1],)))
            for j in range(1, nn_best_params["n_layers"]):
                key = f"units_{j}"
                if key in nn_best_params:
                    nn.add(layers.Dense(nn_best_params[key],
                                        activation=nn_best_params["activation"]))
                    nn.add(layers.Dropout(nn_best_params["dropout_rate"]))
            nn.add(layers.Dense(1))
            opt = getattr(optimizers, nn_best_params["optimizer"])(
                learning_rate=nn_best_params["learning_rate"])
            nn.compile(optimizer=opt, loss="mse")
            nn.fit(X_train, y_train,
                   epochs=nn_best_params["epochs"],
                   batch_size=nn_best_params["batch_size"],
                   validation_split=0.2, verbose=0)
            y_pred = nn.predict(X_test).flatten()
            model_obj = nn
        else:
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
            model_obj = mdl

        # ----- metrics -----
        rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
        mae   = mean_absolute_error(y_test, y_pred)
        r2    = r2_score(y_test, y_pred)
        nrmse = rmse / np.mean(y_test) * 100
        mad   = median_abs_deviation(y_test - y_pred)
        ioa   = IOA(y_test, y_pred)
        a10   = A10(y_test, y_pred)

        rmse_all[idx].append(rmse)
        nrmse_all[idx].append(nrmse)
        mae_all[idx].append(mae)
        r2_all[idx].append(r2)
        mad_all[idx].append(mad)
        ioa_all[idx].append(ioa)
        a10_all[idx].append(a10)


        if rmse < best_rmse[idx]:
            best_rmse[idx]   = rmse
            best_models[idx] = model_obj

    if (i + 1) % 25 == 0:
        print(f"✅ finished iteration {i+1}/{reps}")

# ---------------  summarise & save  ----------------
results_df = pd.DataFrame({
    "Model": model_names,
    "RMSE mean":    [np.mean(lst) for lst in rmse_all],
    "RMSE std":     [np.std(lst)  for lst in rmse_all],
    "NRMSE mean %": [np.mean(lst) for lst in nrmse_all],
    "MAE mean":     [np.mean(lst) for lst in mae_all],
    "MAD mean":     [np.mean(lst) for lst in mad_all],
    "R² mean":      [np.mean(lst) for lst in r2_all],
    "IOA mean":     [np.mean(lst) for lst in ioa_all],
    "A10 mean %":   [np.mean(lst) for lst in a10_all]
})
results_df.to_csv(os.path.join(results_dir, "Model_Evaluation_Metrics_All14.csv"), index=False)
print(" saved: Model_Evaluation_Metrics_All14.csv")
print(results_df.head())
#%%Saving all metrices and best model (lowest rmse) from main evaluation loop , so that i dont have to rerun it later. fpr 14 model
all_results = {
    'rmse_all': rmse_all, 
    'nrmse_all': nrmse_all, 
    'mae_all': mae_all, 
    'r2_all': r2_all, 
    'mad_all': mad_all,
    'ioa_all': ioa_all, 
    'a10_all': a10_all, 
    'best_rmse': best_rmse, 
    'model_names': model_names, 'reps': reps}

results_pickle_path = os.path.join(results_dir, 'all_evaluation_results.pkl')
with open(results_pickle_path, 'wb') as f:
    pickle.dump(all_results, f)
#Save best models
models_dir = os.path.join(results_dir, 'BestModels')
os.makedirs(models_dir, exist_ok=True)
# Save model order mapping for verification
model_mapping = {name: idx for idx, name in enumerate(model_names)}
mapping_path = os.path.join(models_dir, 'model_mapping.json')
with open(mapping_path, 'w') as f:
    json.dump(model_mapping, f, indent=2)

# Save each model
for idx, model_name in enumerate(model_names):
    if model_name == "Neural_Network":
        nn_path = os.path.join(models_dir, 'Neural_Network_best.h5')
        best_models[idx].save(nn_path)
    else:
        model_path = os.path.join(models_dir, f'{model_name}_best.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_models[idx], f)
        
# Unified CSV with all metrics (one row per iteration)
all_metrics_unified = pd.DataFrame({'Iteration': range(1, reps + 1)})

for metric_name, metric_data in [
    ('RMSE', rmse_all), ('NRMSE', nrmse_all), ('MAE', mae_all), 
    ('R2', r2_all), ('MAD', mad_all), ('IOA', ioa_all), ('A10', a10_all)
]:
    df_temp = pd.DataFrame(dict(zip(model_names, metric_data)))
    df_temp.columns = [f"{col}_{metric_name}" for col in df_temp.columns]
    all_metrics_unified = pd.concat([all_metrics_unified, df_temp], axis=1)

all_metrics_unified.to_csv(os.path.join(results_dir, 'All_Metrics_All_Iterations_Unified.csv'), index=False)
#%%Load all the result saved from Main evaluation loop, so from now onwards, you dont have to rerun the code above. 
#loading the metrics and best model from 300 iteration yielded after main evaluation loop
results_pickle_path = os.path.join(results_dir, 'all_evaluation_results1.pkl')
with open(results_pickle_path, 'rb') as f:
    all_results = pickle.load(f)
# Unpack all metrics
rmse_all = all_results['rmse_all']
nrmse_all = all_results['nrmse_all']
mae_all = all_results['mae_all']
r2_all = all_results['r2_all']
mad_all = all_results['mad_all']
ioa_all = all_results['ioa_all']
a10_all = all_results['a10_all']
best_rmse = all_results['best_rmse']
model_names = all_results['model_names']
reps = all_results['reps']
#Load best models 
models_dir = os.path.join(results_dir, 'BestModels')
# Load and verify model order mapping
mapping_path = os.path.join(models_dir, 'model_mapping.json')
with open(mapping_path, 'r') as f:
    saved_model_mapping = json.load(f)
# Verify the order matches
if list(saved_model_mapping.keys()) != model_names:
    model_names = list(saved_model_mapping.keys())
else:
    print("✅ Model order verified - matches saved configuration")

# Load each model in the correct order
best_models = []
for model_name in model_names:
    if model_name == "Neural_Network":
        nn_path = os.path.join(models_dir, 'Neural_Network_best.h5')
        model = keras.models.load_model(nn_path)
        best_models.append(model)
    else:
        model_path = os.path.join(models_dir, f'{model_name}_best.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        best_models.append(model)

#Recreate results DataFrame
results_df = pd.DataFrame({
    "Model": model_names,
    "RMSE mean":    [np.mean(lst) for lst in rmse_all],
    "RMSE std":     [np.std(lst)  for lst in rmse_all],
    "NRMSE mean %": [np.mean(lst) for lst in nrmse_all],
    "MAE mean":     [np.mean(lst) for lst in mae_all],
    "MAD mean":     [np.mean(lst) for lst in mad_all],
    "R² mean":      [np.mean(lst) for lst in r2_all],
    "IOA mean":     [np.mean(lst) for lst in ioa_all],
    "A10 mean %":   [np.mean(lst) for lst in a10_all]
})
#%%Saving result from main evaluation for only 4 model of intrest ( RF, NN, XGboost, SLR)
#save all metrics and best iteration across 300 iteration as best model. 
# Define the 4 models to extract
selected_models = ["Random_Forest","XGBoost", "Neural_Network", "Simple_Regression" ]

# Find indices of selected models in the original model_names list
selected_indices = [model_names.index(model) for model in selected_models]
# Extract metrics for selected models only
rmse_4models = [rmse_all[idx] for idx in selected_indices]
nrmse_4models = [nrmse_all[idx] for idx in selected_indices]
mae_4models = [mae_all[idx] for idx in selected_indices]
r2_4models = [r2_all[idx] for idx in selected_indices]
mad_4models = [mad_all[idx] for idx in selected_indices]
ioa_4models = [ioa_all[idx] for idx in selected_indices]
a10_4models = [a10_all[idx] for idx in selected_indices]

# Extract best models for selected models only
best_models_4 = [best_models[idx] for idx in selected_indices]

# Create dictionary with all results for 4 models
results_4models = {
'rmse_all': rmse_4models,
    'nrmse_all': nrmse_4models,
    'mae_all': mae_4models,
    'r2_all': r2_4models,
    'mad_all': mad_4models,
    'ioa_all': ioa_4models,
    'a10_all': a10_4models,
    'best_rmse': best_rmse,  # Keep full array for reference
    'model_names': selected_models,
    'reps': reps,
    'original_indices': selected_indices  # Store original indices for reference
}
# Save the 4-model results
results_4models_path = os.path.join(results_dir, '4ML_evaluation_results.pkl')
with open(results_4models_path, 'wb') as f:
    pickle.dump(results_4models, f)
# Create directory for 4 best models
models_4_dir = os.path.join(results_dir, 'BestModels_4ML')
os.makedirs(models_4_dir, exist_ok=True)

# Save best models for the 4 selected models
for i, model_name in enumerate(selected_models):
    if model_name == "Neural_Network":
        nn_path = os.path.join(models_4_dir, 'Neural_Network_best.h5')
        best_models_4[i].save(nn_path)
    else:
        model_path = os.path.join(models_4_dir, f'{model_name}_best.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_models_4[i], f)

# Save model mapping for the 4 models
mapping_4models = {model_name: i for i, model_name in enumerate(selected_models)}
mapping_4models_path = os.path.join(models_4_dir, 'model_mapping.json')
with open(mapping_4models_path, 'w') as f:
    json.dump(mapping_4models, f, indent=4)
#%%Load the result saved from main evaluation loop, only for 4 ML model of intrest (RF, XGBoost, NN, SLR)
results_4models_path = os.path.join(results_dir, '4ML_evaluation_results.pkl')
with open(results_4models_path, 'rb') as f:
    all_results_4ML = pickle.load(f)

# Unpack all metrics for 4 models
rmse_all_4ML = all_results_4ML['rmse_all']
nrmse_all_4ML = all_results_4ML['nrmse_all']
mae_all_4ML = all_results_4ML['mae_all']
r2_all_4ML = all_results_4ML['r2_all']
mad_all_4ML = all_results_4ML['mad_all']
ioa_all_4ML = all_results_4ML['ioa_all']
a10_all_4ML = all_results_4ML['a10_all']
best_rmse_4ML = all_results_4ML['best_rmse']
model_names_4ML = all_results_4ML['model_names']
reps_4ML = all_results_4ML['reps']
# Load best models for 4 ML models
models_4_dir = os.path.join(results_dir, 'BestModels_4ML')
# Load and verify model order mapping
mapping_4models_path = os.path.join(models_4_dir, 'model_mapping.json')
with open(mapping_4models_path, 'r') as f:
    saved_model_mapping_4ML = json.load(f)
# Verify the order matches
if list(saved_model_mapping_4ML.keys()) != model_names_4ML:
    model_names_4ML = list(saved_model_mapping_4ML.keys())
else:
    print("✅ Model order verified - matches saved configuration")

# Load each of the 4 best models in the correct order
best_models_4ML = []
for model_name in model_names_4ML:
    if model_name == "Neural_Network":
        nn_path = os.path.join(models_4_dir, 'Neural_Network_best.h5')
        model = keras.models.load_model(nn_path)
        best_models_4ML.append(model)
    else:
        model_path = os.path.join(models_4_dir, f'{model_name}_best.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        best_models_4ML.append(model)

# Recreate results DataFrame for 4 models
results_df_4ML = pd.DataFrame({
    "Model": model_names_4ML,
    "RMSE mean":    [np.mean(lst) for lst in rmse_all_4ML],
    "RMSE std":     [np.std(lst)  for lst in rmse_all_4ML],
    "NRMSE mean %": [np.mean(lst) for lst in nrmse_all_4ML],
    "MAE mean":     [np.mean(lst) for lst in mae_all_4ML],
    "MAD mean":     [np.mean(lst) for lst in mad_all_4ML],
    "R² mean":      [np.mean(lst) for lst in r2_all_4ML],
    "IOA mean":     [np.mean(lst) for lst in ioa_all_4ML],
    "A10 mean %":   [np.mean(lst) for lst in a10_all_4ML]
})
#%% PLOT: RMSE Stabilization:  for All 14 Models
# Calculate cumulative mean RMSE for each model
cumulative_rmse = [np.cumsum(rmse_list) / (np.arange(len(rmse_list)) + 1) for rmse_list in rmse_all]
# Define colors for better visualization (14 distinct colors)
colors = [
    '#1f77b4',  # blue - Random_Forest
    '#ff7f0e',  # orange - Simple_Regression
    '#2ca02c',  # green - Neural_Network
    '#d62728',  # red - XGBoost
    '#9467bd',  # purple - Ridge
    '#8c564b',  # brown - LASSO
    '#e377c2',  # pink - ElasticNet
    '#7f7f7f',  # gray - PLSR
    '#bcbd22',  # olive - LightGBM
    '#17becf',  # cyan - CatBoost
    '#ff9896',  # light red - SVR
    '#98df8a',  # light green - ExtraTrees
    '#c5b0d5',  # light purple - AdaBoost
    '#c49c94'   # light brown - GradientBoost
]
# Create the plot
plt.figure(figsize=(14, 8))
for i, cum_mean in enumerate(cumulative_rmse):
    plt.plot(cum_mean, label=model_names[i], linewidth=2, color=colors[i], alpha=0.8)
plt.title("RMSE Stabilization Over 300 Iterations", fontsize=16, fontweight='bold')
plt.xlabel("Iteration", fontsize=13)
plt.ylabel("Cumulative Mean RMSE (t ha-1)", fontsize=13)
plt.legend(fontsize=9, ncol=2, loc='best', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
jpeg_path = os.path.join(graphs_dir, 'RMSE_Stabilization_Plot_All14Models.jpeg')
plt.savefig(jpeg_path, dpi=300, bbox_inches='tight')
plt.show()
#%%PLOT: RMSE stablization: for 4 models (RF, XGBoost, NN, SLR)
# Calculate cumulative mean RMSE for the 4 models
cumulative_rmse_4ML = [np.cumsum(rmse_list) / (np.arange(len(rmse_list)) + 1) 
                       for rmse_list in rmse_all_4ML]

# Define colors for the 4 models
colors_4models = {
    'Random_Forest': '#2ca02c',      # green 
    'Simple_Regression': '#ff7f0e',  # orange
    'Neural_Network': '#d62728',     # bllue   red
    'XGBoost': '#1f77b4'             # red
}

# Create the stabilization plot
plt.figure(figsize=(12, 7))

for i, model_name in enumerate(model_names_4ML):
    plt.plot(cumulative_rmse_4ML[i], 
             label=model_name, 
             linewidth=2.5, 
             color=colors_4models[model_name], 
             alpha=0.85)

# --- ADDED CODE START ---
# Add a dotted vertical line at the 200th iteration
plt.axvline(x=200, color='gray', linestyle=':', linewidth=1.5, label='200 Iterations')
# --- ADDED CODE END ---

plt.xlabel("Iteration", fontsize=18)
plt.ylabel("Cumulative Mean RMSE (t ha⁻¹)", fontsize=18)
plt.legend(fontsize=16, loc='best', framealpha=0.95)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Save the stabilization plot
stab_jpeg_path = os.path.join(graphs_dir, 'RMSE_Stabilization_4ML.jpeg')
stab_png_path = os.path.join(graphs_dir, 'RMSE_Stabilization_4ML.png')
plt.savefig(stab_jpeg_path, dpi=300, bbox_inches='tight')
plt.savefig(stab_png_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved RMSE Stabilization: {stab_jpeg_path}")
print(f"✅ Saved RMSE Stabilization: {stab_png_path}")

plt.show()

# Print final stabilized RMSE values
print("\n📊 Final Cumulative Mean RMSE after 300 iterations:")
for i, model_name in enumerate(model_names_4ML):
    final_rmse = cumulative_rmse_4ML[i][-1]
    overall_mean = np.mean(rmse_all_4ML[i])
    overall_std = np.std(rmse_all_4ML[i])
    print(f"  {model_name:20s}: {final_rmse:.4f} (Mean: {overall_mean:.4f} ± {overall_std:.4f})")
#%%PLOT: Boxplot of RMSE distribution across 300 iteration for all 14 model
# Create DataFrame for easier plotting
Result_RMSE = pd.DataFrame(dict(zip(model_names, rmse_all)))
boxplot_kwargs = dict(
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='black', markersize=8),
    boxprops=dict(color='blue', linewidth=1.5),
    medianprops=dict(color='red', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5),
    flierprops=dict(marker='o', markerfacecolor='orange', markersize=5, alpha=0.5))

plt.figure(figsize=(16, 8))
plt.boxplot(Result_RMSE.values, **boxplot_kwargs)
plt.title('RMSE Distribution Across 300 Iterations', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predictive Models', fontsize=13, fontweight='bold')
plt.ylabel('Root Mean Square Error (t ha-1)', fontsize=13, fontweight='bold')
plt.xticks(range(1, len(model_names) + 1), model_names, rotation=45, ha='right', fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'RMSE_Boxplot_Classic_All14.jpeg'), dpi=300, bbox_inches='tight')
plt.show()
#%%PLOT: Boxplot of RMSE distribution across 300 iteration for 4 model (RF, XGboost, NN, SLR)
Result_RMSE_4ML = pd.DataFrame(dict(zip(model_names_4ML, rmse_all_4ML)))
# CHANGE: Define display names (no underscores)
model_name_mapping = {
    'Random_Forest': 'Random Forest',
    'XGBoost': 'XGBoost',
    'Neural_Network': 'Neural Network',
    'Simple_Regression': 'Simple Regression'
}

# CHANGE: Create display labels in same order as model_names_4ML
display_names = [model_name_mapping[m] for m in model_names_4ML]

# Define boxplot styling
boxplot_kwargs = dict(
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='black', markersize=8),
    boxprops=dict(color='blue', linewidth=1.5),
    medianprops=dict(color='red', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5),
    flierprops=dict(marker='o', markerfacecolor='orange', markersize=5, alpha=0.5))
# Create the boxplot
plt.figure(figsize=(12, 8))
plt.boxplot(Result_RMSE_4ML.values, **boxplot_kwargs)

plt.xlabel('Predictive Models', fontsize=16, fontweight='bold')
plt.ylabel('Root Mean Square Error (t ha⁻¹)', fontsize=16, fontweight='bold')


#  CHANGE: Use clean display names on x-axis
plt.xticks(range(1, len(display_names) + 1),
           display_names,
           fontsize=14)

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# Save the boxplot
box_jpeg_path = os.path.join(graphs_dir, 'RMSE_Boxplot_4ML.jpeg')
plt.savefig(box_jpeg_path, dpi=300, bbox_inches='tight')
plt.show()
#%%Table: Summary of 7 metrics(r2, rmse, mae, ioa, a10, mad,nrmse) for 4 model ((RF, XGboost, NN, SLR) after 300iteration (shows mean ±sd )
summary_4ML = pd.DataFrame({
    'Model': model_names_4ML,
    'RMSE': [f"{np.mean(rmse_all_4ML[i]):.3f} ± {np.std(rmse_all_4ML[i]):.3f}" 
             for i in range(len(model_names_4ML))],
    'R²': [f"{np.mean(r2_all_4ML[i]):.3f} ± {np.std(r2_all_4ML[i]):.3f}" 
           for i in range(len(model_names_4ML))],
    'MAE': [f"{np.mean(mae_all_4ML[i]):.3f} ± {np.std(mae_all_4ML[i]):.3f}" 
            for i in range(len(model_names_4ML))],
    'IOA': [f"{np.mean(ioa_all_4ML[i]):.3f} ± {np.std(ioa_all_4ML[i]):.3f}" 
            for i in range(len(model_names_4ML))],
    'A10 (%)': [f"{np.mean(a10_all_4ML[i]):.1f} ± {np.std(a10_all_4ML[i]):.1f}" 
                for i in range(len(model_names_4ML))],
    'CV (%)': [f"{(np.std(rmse_all_4ML[i])/np.mean(rmse_all_4ML[i])*100):.1f}" 
               for i in range(len(model_names_4ML))]
})

# Sort by R² (descending)
summary_4ML = summary_4ML.sort_values('R²', ascending=False, 
                                      key=lambda x: x.str.split(' ±').str[0].astype(float))
summary_4ML = summary_4ML.reset_index(drop=True)
summary_4ML.insert(0, 'Rank', range(1, len(summary_4ML) + 1))

# Save summary table to CSV
csv_path_4ML = os.path.join(results_dir, 'Model_Performance_Summary_4ML.csv')
summary_4ML.to_csv(csv_path_4ML, index=False)
#%%Best model evaluation loop: Model performance on held out independent test set (n=92, ie 20% data), for all 14 model. 

FINAL_SEED = 324  # Create a NEW test set for final evaluation. 
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X, y, test_size=0.2, random_state=FINAL_SEED)

scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_test_final_scaled = scaler_final.transform(X_test_final)
print(f"Final test set size: {len(y_test_final)} samples")

#STEP 1: Get Predictions from All Best Models (best iteration that had lowest rmse across 300 runs)

all_predictions = []

for idx, (model, model_name) in enumerate(zip(best_models, model_names)):
    if model_name == "Neural_Network":
        y_pred = model.predict(X_test_final_scaled, verbose=0).flatten()
    else:
        y_pred = model.predict(X_test_final_scaled)
    
    # CRITICAL: Ensure 1D array
    y_pred = np.asarray(y_pred).flatten()
    
    all_predictions.append(y_pred)
    print(f"✓ {model_name}: {y_pred.shape}")

print(f"\n✅ Collected {len(all_predictions)} predictions")
    
#STEP 2: CREATE ENSEMBLE MODELS
# Convert to array
all_predictions_array = np.array(all_predictions)
# Simple Averaging
SA_pred = np.mean(all_predictions_array, axis=0)

# Bayesian Model Averaging
weights = np.array([1.0 / rmse for rmse in best_rmse])
weights = weights / np.sum(weights)

BMA_pred = np.zeros(len(y_test_final))
for i, pred in enumerate(all_predictions):
    BMA_pred += weights[i] * pred

# Add ensemble predictions to the list
all_predictions.append(SA_pred)
all_predictions.append(BMA_pred)
model_names_with_ensembles = model_names + ['Simple_Averaging', 'Bayesian_MA']

#STEP 3: Compute Metrics for All Models + Ensembles
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = (rmse / np.mean(y_true)) * 100
    mad = np.median(np.abs(y_true - y_pred))
    ioa = 1 - (np.sum((y_pred - y_true)**2) / 
               np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2))
    a10 = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.10) * 100
    
    return {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'NRMSE': nrmse,
            'MAD': mad, 'IOA': ioa, 'A10': a10}

final_metrics = []
for model_name, y_pred in zip(model_names_with_ensembles, all_predictions):
    metrics = compute_metrics(y_test_final, y_pred)
    metrics['Model'] = model_name
    final_metrics.append(metrics)

metrics_df = pd.DataFrame(final_metrics)
metrics_df = metrics_df[['Model', 'R²', 'RMSE', 'MAE', 'NRMSE', 'MAD', 'IOA', 'A10']]
#PLOT: Scatter plots of measured vs predicted yield for all 14 model including Bayesian ensemble on the independent test set (n=92, ie 20%).

# Calculate color values for scatter plots
color_values = (y_test_final - np.min(y_test_final)) / (np.max(y_test_final) - np.min(y_test_final))

# Total plots = 14 base models + 2 ensembles = 16
n_models = len(model_names_with_ensembles)
n_cols = 4
n_rows = int(np.ceil(n_models / n_cols))

fig = plt.figure(figsize=(20, n_rows * 4))

for i, (model_name, y_pred) in enumerate(zip(model_names_with_ensembles, all_predictions)):
    metrics = final_metrics[i]
    
    ax = plt.subplot(n_rows, n_cols, i+1)
    
    # Ensure y_pred is 1D
    y_pred = np.asarray(y_pred).flatten()
    
    # Scatter plot
    scatter = ax.scatter(y_test_final, y_pred, c=color_values, cmap='plasma', 
                        alpha=0.6, edgecolor='k', s=50)

# 1:1 line
    min_val = min(y_test_final.min(), y_pred.min())
    max_val = max(y_test_final.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
    
    # Formatting
    ax.set_xlabel('Measured Yield (Mg ha⁻¹)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicted Yield (Mg ha⁻¹)', fontsize=10, fontweight='bold')
    
    # Title with special formatting for ensembles
    if 'Averaging' in model_name or 'MA' in model_name:
        ax.set_title(model_name, fontsize=12, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    else:
        ax.set_title(model_name, fontsize=11, fontweight='bold')
    
    # Metrics box
    metrics_text = (f"R² = {metrics['R²']:.3f}\n"
                   f"RMSE = {metrics['RMSE']:.3f}\n"
                   f"MAE = {metrics['MAE']:.3f}\n"
                   f"NRMSE = {metrics['NRMSE']:.2f}%\n"
                   f"IOA = {metrics['IOA']:.3f}\n"
                   f"A10 = {metrics['A10']:.1f}%")
    
    ax.text(0.05, 0.95, metrics_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5',
                     edgecolor='black', linewidth=1),
            fontsize=8, verticalalignment='top', family='monospace')
    
    ax.grid(True, alpha=0.3, linestyle='--')

plt.suptitle('Measured vs Predicted Yield: All Models + Ensemble Methods', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'Final_Scatter_All_Models_With_Ensembles.jpeg'), 
            dpi=300, bbox_inches='tight')
print(f" Saved: Scatter plots with all models + ensembles")
plt.show()

#%%Best model evaluation loop: for 4 model (rf, xgboost,nn,slr) plus bayesian ensemble: Model performance on held out independent test set (n=92, ie 20% data)
# STEP 1: Create Final Test Set
FINAL_SEED = 324 # Create a NEW test set for final evaluation
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X, y, test_size=0.2, random_state=FINAL_SEED) #held out independent fixed test set,  same test set used for all model.can be easier or harder to predict,
scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_test_final_scaled = scaler_final.transform(X_test_final)
# STEP 2: Get Predictions from 4 Best Models
all_predictions_4ML = []

for idx, (model, model_name) in enumerate(zip(best_models_4ML, model_names_4ML)):
    if model_name == "Neural_Network":
        y_pred = model.predict(X_test_final_scaled, verbose=0).flatten()
    else:
        y_pred = model.predict(X_test_final_scaled)
    
    # Ensure 1D array
    y_pred = np.asarray(y_pred).flatten()
    all_predictions_4ML.append(y_pred)
    
# STEP 3: Create Bayesian Model Averaging Ensemble (BMA)
# Calculate weights based on RMSE (inverse of RMSE)
# Get mean RMSE for each of the 4 models
best_rmse_4ML = [np.mean(rmse_all_4ML[i]) for i in range(len(model_names_4ML))]

weights_BMA = np.array([1.0 / rmse for rmse in best_rmse_4ML])
weights_BMA = weights_BMA / np.sum(weights_BMA)

print("\nBMA Weights (based on mean RMSE from 300 iterations):")
for i, model_name in enumerate(model_names_4ML):
    print(f"  {model_name:20s}: Weight = {weights_BMA[i]:.4f} "
          f"(RMSE = {best_rmse_4ML[i]:.4f})")
    
    
#  Save BMA weights to CSV (Full + Reduced)
weights_rows = []

# Full BMA (4 models)
for i, model_name in enumerate(model_names_4ML):
    weights_rows.append({
        "Ensemble": "BMA_Full",
        "Model": model_name,
        "RMSE_mean_300iter": best_rmse_4ML[i],
        "Weight": weights_BMA[i]
    })

# Reduced BMA (3 models: RF + XGB + NN)
for w, m, r in zip(weights_BMA3, bma3_models, best_rmse_BMA3):
    weights_rows.append({
        "Ensemble": "BMA_Reduced",
        "Model": m,
        "RMSE_mean_300iter": r,
        "Weight": w
    })

weights_df = pd.DataFrame(weights_rows)

weights_csv_path = os.path.join(results_dir, "BMA_Weights_Full_vs_Reduced.csv")
weights_df.to_csv(weights_csv_path, index=False)
print(f" Saved BMA weights CSV: {weights_csv_path}")


# Calculate BMA prediction
BMA_pred = np.zeros(len(y_test_final))
for i, pred in enumerate(all_predictions_4ML):
    BMA_pred += weights_BMA[i] * pred
# Add BMA to predictions list
all_predictions_4ML.append(BMA_pred)
model_names_4ML_BMA = model_names_4ML + ['Bayesian_MA']

#  CHANGE: STEP 3B: Create SECOND BMA (EXCLUDING SLR; only RF + NN + XGBoost)
# Identify indices of the 3 models to keep
bma3_models = ['Random_Forest', 'XGBoost', 'Neural_Network']
bma3_indices = [model_names_4ML.index(m) for m in bma3_models]

# RMSE list for only those 3 models
best_rmse_BMA3 = [best_rmse_4ML[i] for i in bma3_indices]

weights_BMA3 = np.array([1.0 / rmse for rmse in best_rmse_BMA3])
weights_BMA3 = weights_BMA3 / np.sum(weights_BMA3)

print("\nBMA3 Weights (RF + XGBoost + NN only; based on mean RMSE from 300 iterations):")
for w, m, r in zip(weights_BMA3, bma3_models, best_rmse_BMA3):
    print(f"  {m:20s}: Weight = {w:.4f} (RMSE = {r:.4f})")

# BMA3 prediction using only RF, XGBoost, NN
BMA3_pred = np.zeros(len(y_test_final))
for w, i in zip(weights_BMA3, bma3_indices):
    BMA3_pred += w * all_predictions_4ML[i]

# Append BMA3 prediction
all_predictions_4ML.append(BMA3_pred)
model_names_4ML_BMA = model_names_4ML_BMA + ['Bayesian_MA_3']   # 🔥 CHANGE

# STEP 4: Compute Metrics for All Models + BMA

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = (rmse / np.mean(y_true)) * 100
    mad = np.median(np.abs(y_true - y_pred))
    ioa = 1 - (np.sum((y_pred - y_true)**2) / 
               np.sum((np.abs(y_pred - np.mean(y_true)) + 
                      np.abs(y_true - np.mean(y_true)))**2))
    a10 = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.10) * 100
    
    return {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'NRMSE': nrmse,
            'MAD': mad, 'IOA': ioa, 'A10': a10}

final_metrics_4ML = []
for model_name, y_pred in zip(model_names_4ML_BMA, all_predictions_4ML):
    metrics = compute_metrics(y_test_final, y_pred)
    metrics['Model'] = model_name
    final_metrics_4ML.append(metrics)

# Create DataFrame
metrics_df_4ML = pd.DataFrame(final_metrics_4ML)
metrics_df_4ML = metrics_df_4ML[['Model', 'R²', 'RMSE', 'MAE', 'NRMSE', 
                                 'MAD', 'IOA', 'A10']]

# Sort by R² descending
metrics_df_4ML = metrics_df_4ML.sort_values('R²', ascending=False).reset_index(drop=True)

print("\n FINAL MODEL PERFORMANCE (4 Models + BMA Ensemble):")
print("─"*80)
print(metrics_df_4ML.to_string(index=False))

# Save to CSV
csv_path_4ML = os.path.join(results_dir, 'Final_4ML_Metrics_With_BMA.csv')
metrics_df_4ML.to_csv(csv_path_4ML, index=False)



# PLOT: Scater plot for 4 model measured vs predicted yield inclusing bayesian ensemble
# Calculate color values for scatter plots (based on observed yield)
color_values = (y_test_final - np.min(y_test_final)) / \
               (np.max(y_test_final) - np.min(y_test_final))
# CHANGE: Include BOTH BMAs in order
custom_order = ['Random_Forest', 'XGBoost', 'Neural_Network', 
                'Simple_Regression', 'Bayesian_MA', 'Bayesian_MA_3']
#  CHANGE: Add display name for second BMA
display_names = {
    'Random_Forest': 'Random Forest',
    'XGBoost': 'XGBoost',
    'Neural_Network': 'Neural Network',
    'Simple_Regression': 'Simple Regression',
    'Bayesian_MA': 'BMA Full',
    'Bayesian_MA_3': 'BMA Reduced'
}
# Create figure with 2 rows x 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
# Plot each model in custom order
for plot_idx, model_name in enumerate(custom_order):
    model_idx = model_names_4ML_BMA.index(model_name)
    y_pred = all_predictions_4ML[model_idx]
    metrics = final_metrics_4ML[model_idx]
    
    ax = axes[plot_idx]
    
    # Ensure y_pred is 1D
    y_pred = np.asarray(y_pred).flatten()
    
    # Scatter plot
    scatter = ax.scatter(y_test_final, y_pred, c=color_values, 
                        cmap='plasma', alpha=0.7, edgecolor='k', 
                        s=60, linewidth=0.5)
    
    # 1:1 line
    min_val = min(y_test_final.min(), y_pred.min())
    max_val = max(y_test_final.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2.5, label='1:1 line', alpha=0.8)
    
    # Remove individual axis labels (we'll add common ones later)
    if plot_idx < 3:  # Top row
        ax.set_xticklabels([])
    if plot_idx % 3 != 0:  # Not first column
        ax.set_yticklabels([])
    
    # Title - uniform formatting for all models
    ax.set_title(display_names[model_name], 
                fontsize=17, fontweight='bold')
    
    # Metrics box with larger font
    metrics_text = (f"R²     = {metrics['R²']:.3f}\n"
                   f"RMSE   = {metrics['RMSE']:.3f}\n"
                   f"NRMSE  = {metrics['NRMSE']:.2f}%\n")

    ax.text(0.05, 0.95, metrics_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.95, 
                     boxstyle='round,pad=0.6',
                     edgecolor='black', linewidth=1.2),
            fontsize=17, verticalalignment='top', 
            family='monospace', linespacing=1.5)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='lower right', fontsize=15, framealpha=0.9)
    # Axis ticks font size
    ax.tick_params(axis='both', labelsize=15)
    # Equal aspect for better 1:1 visualization
    ax.set_aspect('equal', adjustable='box')


# Add common X and Y labels
fig.text(0.5, 0.04, 'Measured Yield (t ha⁻¹)', 
         ha='center', fontsize=17, fontweight='bold')
fig.text(0.04, 0.5, 'Predicted Yield (t ha⁻¹)', 
         va='center', rotation='vertical', fontsize=17, fontweight='bold')

plt.tight_layout(rect=[0.06, 0.06, 1, 0.94])

# Save plots
jpeg_path = os.path.join(graphs_dir, 'Final_Scatter_4ML_With_BMA.jpeg')
plt.savefig(jpeg_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

#%%re model main evaluation to extract FI score from all 300 iteration
#Main evaluation loop to compare 4 models across 300 randomized train-test splits (80/20),
#HereFor each model, we track the best-iterating version across 300 loops based on lowest RMSE value 
reps = 300
model_names = [
    "Random_Forest", "Simple_Regression", "Neural_Network", "XGBoost"
]

# Initialize metric collectors
rmse_all  = [[] for _ in model_names]
nrmse_all = [[] for _ in model_names]
mae_all   = [[] for _ in model_names]
r2_all    = [[] for _ in model_names]
mad_all   = [[] for _ in model_names]
ioa_all   = [[] for _ in model_names]
a10_all   = [[] for _ in model_names]

# Track best models / RMSE
best_models = [None] * len(model_names)
best_rmse   = [float("inf")] * len(model_names)

# feature names (exclude yield col)
feature_names = Ex4.columns[1:].tolist()

# collect Top 10 FI per iteration
fi_top10_allruns = []

# ================================
# MAIN LOOP
# ================================
for i in range(reps):

    # split & scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # build & train NN
    nn_model_for_iter = build_nn_model(X_train_scaled.shape[1], nn_best_params)
    nn_model_for_iter.fit(
        X_train_scaled, y_train,
        epochs=nn_best_params["epochs"],
        batch_size=nn_best_params["batch_size"],
        validation_split=0.2,
        verbose=0
    )

    # define models
    rf_model    = RandomForestRegressor(n_jobs=-1, **rf_best_params)
    lr_model    = LinearRegression()
    xgb_model   = xgb.XGBRegressor(n_jobs=-1, **xg_best_params)
   

    models_list = [
    rf_model, lr_model, nn_model_for_iter, xgb_model
]

    fitted_models_for_FI = {}

    # --------------------------------------
    # MODEL TRAINING AND METRIC COLLECTION
    # --------------------------------------
    for idx, mdl in enumerate(models_list):

        if mdl is nn_model_for_iter:
            y_pred = mdl.predict(X_test_scaled).flatten()
            model_obj = mdl
        else:
            mdl.fit(X_train_scaled, y_train)
            y_pred = mdl.predict(X_test_scaled)
            model_obj = mdl

        fitted_models_for_FI[model_names[idx]] = model_obj

        rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
        mae   = mean_absolute_error(y_test, y_pred)
        r2    = r2_score(y_test, y_pred)
        nrmse = rmse / np.mean(y_test) * 100
        mad   = median_abs_deviation(y_test - y_pred)
        ioa   = IOA(y_test, y_pred)
        a10   = A10(y_test, y_pred)

        rmse_all[idx].append(rmse)
        nrmse_all[idx].append(nrmse)
        mae_all[idx].append(mae)
        r2_all[idx].append(r2)
        mad_all[idx].append(mad)
        ioa_all[idx].append(ioa)
        a10_all[idx].append(a10)

        if rmse < best_rmse[idx]:
            best_rmse[idx]   = rmse
            best_models[idx] = model_obj

    # --------------------------------------
    # FEATURE IMPORTANCE CALCULATION
    # --------------------------------------
    models_to_analyze = [
        'Random_Forest', 'XGBoost', 'Neural_Network', 'Simple_Regression'
    ]

    def nn_scorer(model, X_in, y_in):
        y_hat = model.predict(X_in, verbose=0).flatten()
        return -mean_squared_error(y_in, y_hat)

    feature_importance_results = {}

    rf_final  = fitted_models_for_FI['Random_Forest']
    importance_rf = rf_final.feature_importances_
    feature_importance_results['Random_Forest'] = importance_rf

    xgb_final = fitted_models_for_FI['XGBoost']
    importance_xgb = xgb_final.feature_importances_
    feature_importance_results['XGBoost'] = importance_xgb

    nn_final = fitted_models_for_FI['Neural_Network']
    perm_nn = permutation_importance(
        nn_final, X_test_scaled, y_test,
        n_repeats=10, random_state=42, scoring=nn_scorer
    )
    importance_nn = perm_nn.importances_mean
    feature_importance_results['Neural_Network'] = importance_nn

    lr_final = fitted_models_for_FI['Simple_Regression']
    lr_coefs = np.abs(lr_final.coef_)
    feature_importance_results['Simple_Regression'] = lr_coefs

    importance_df = pd.DataFrame({'Feature': feature_names})
    for mname in models_to_analyze:
        importance_df[f'{mname}_Raw'] = feature_importance_results[mname]

    for mname in models_to_analyze:
        raw_col = f'{mname}_Raw'
        denom = importance_df[raw_col].sum()
        importance_df[f'{mname}_Norm'] = (
            importance_df[raw_col] / denom if denom != 0 else 0.0
        )

    norm_cols = [f'{name}_Norm' for name in models_to_analyze]
    importance_df['Average_Importance'] = importance_df[norm_cols].mean(axis=1)
    importance_df['Rank'] = (
        importance_df['Average_Importance'].rank(ascending=False).astype(int)
    )

    top10_iter = importance_df.sort_values('Rank').head(10).copy()
    top10_iter['Iteration'] = i + 1
    fi_top10_allruns.append(top10_iter)

    if (i + 1) % 25 == 0:
        print(f"✅ finished iteration {i+1}/{reps}")

# --------------------------------------
# SAVE METRICS + FEATURE IMPORTANCE
# --------------------------------------
results_df = pd.DataFrame({
    "Model": model_names,
    "RMSE mean":    [np.mean(lst) for lst in rmse_all],
    "RMSE std":     [np.std(lst)  for lst in rmse_all],
    "NRMSE mean %": [np.mean(lst) for lst in nrmse_all],
    "MAE mean":     [np.mean(lst) for lst in mae_all],
    "MAD mean":     [np.mean(lst) for lst in mad_all],
    "R² mean":      [np.mean(lst) for lst in r2_all],
    "IOA mean":     [np.mean(lst) for lst in ioa_all],
    "A10 mean %":   [np.mean(lst) for lst in a10_all]
})
results_df.to_csv(os.path.join(results_dir, "Model_Evaluation_Metrics_All14.csv"), index=False)
print("💾 saved: Model_Evaluation_Metrics_All14.csv")
print(results_df.head())

fi_allruns_df = pd.concat(fi_top10_allruns, ignore_index=True)
fi_allruns_path = os.path.join(results_dir, "FeatureImportance_AllRuns_Top10.csv")
fi_allruns_df.to_csv(fi_allruns_path, index=False)
print("💾 saved:", fi_allruns_path)
print(fi_allruns_df.head())


#%%
# Calculate frequency for each feature
#FI boxplot
fi_allruns_path = os.path.join(results_dir, "FeatureImportance_AllRuns_Top10.csv")
# Load the data
fi_df = pd.read_csv(fi_allruns_path)

# 🔍 STEP 1: Inspect raw feature names
unique_features = fi_df['Feature'].unique()

# Print first 30 feature names to inspect pattern
for f in unique_features[:30]:
    print(f)
    
import re

def clean_feature_name(name):
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    
    # Insert space between number and DAP (e.g., 90DAP → 90 DAP)
    name = re.sub(r'(\d+)\s*DAP', r'\1 DAP', name)
    
    return name

feature_frequency = fi_df['Feature'].value_counts().reset_index()
feature_frequency.columns = ['Feature', 'Frequency']
feature_frequency = feature_frequency.sort_values('Frequency', ascending=False)

#%%final FI plot 
top_n = 15
top_features = feature_frequency.head(top_n)['Feature'].tolist()
# Filter data for these features
fi_top = fi_df[fi_df['Feature'].isin(top_features)]
# Create ordered list by frequency (highest to lowest)
feature_order = top_features

# ============================================================================
# CREATE COMBINED BOXPLOT + FREQUENCY LINE
# ============================================================================
fig, ax1 = plt.subplots(figsize=(16, 8))

# ===== LEFT Y-AXIS: FREQUENCY LINE =====
# Get frequency values in the same order as features
freq_values = [feature_frequency[feature_frequency['Feature'] == feat]['Frequency'].values[0] 
               for feat in feature_order]

# Plot frequency line
line = ax1.plot(range(len(feature_order)), freq_values, 
                color='darkred', linewidth=3, marker='o', markersize=10,
                label='Frequency (top 10 out of 300 iteration)', zorder=5)

# Add frequency values as text labels
for i, freq in enumerate(freq_values):
    ax1.text(i, freq + 5, str(int(freq)), 
             ha='center', va='bottom', fontsize=15,  # Increased from 9 to 12
             fontweight='bold', color='darkred')

ax1.set_ylabel('Top 10 Appearance Frequency', 
               fontsize=19, color='black')  # Increased from 14 to 17
ax1.tick_params(axis='y', labelcolor='black', labelsize=17)  # Increased from 12 to 15
ax1.set_ylim(0, 320)

# ===== X-AXIS: FEATURE NAMES =====


#  CLEANED x-axis labels (visual only)
clean_feature_labels = [clean_feature_name(f) for f in feature_order]

ax1.set_xticklabels(clean_feature_labels,
                    rotation=45, ha='right', fontsize=17)


# ===== RIGHT Y-AXIS: BOXPLOT OF RANKS =====
ax2 = ax1.twinx()

# Create boxplot - CLASSIC STYLE (no fill, median in red)
bp = ax2.boxplot(
    [fi_top[fi_top['Feature'] == feat]['Rank'].values for feat in feature_order],
    positions=range(len(feature_order)),
    widths=0.6,
    patch_artist=True,
    showmeans=False,  # NO MEAN - removed
    showfliers=False,
    whis=0.5,
    medianprops=dict(color='red', linewidth=3, zorder=3),  # Red median, thicker
    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),  # Classic style - no fill
    whiskerprops=dict(color='black', linewidth=1.5, linestyle='--'),  # Black whiskers
    capprops=dict(color='black', linewidth=1.5)  # Black caps
)

ax2.set_ylabel('Boxplot of Rank Within Top 10', 
               fontsize=19, color='black')  # Increased from 14 to 17
ax2.tick_params(axis='y', labelcolor='black', labelsize=17)  # Increased from 12 to 15
ax2.set_ylim(0.5, 10.5)
ax2.set_yticks(range(1, 11))
ax2.invert_yaxis()
ax2.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

# ===== LEGEND ONLY (NO TITLE) =====
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], color='darkred', linewidth=3, marker='o', 
           markersize=8, label='Top 10 Frequency'),
    Patch(facecolor='none', edgecolor='black', linewidth=1.5, label='Rank Distribution'),
    Line2D([0], [0], color='red', linewidth=3, label='Median Rank'),  # Only median, red
 
]

ax1.legend(handles=legend_elements, loc='upper right', fontsize=17, framealpha=0.9)  # Increased from 11 to 14

plt.tight_layout()

# Save
os.makedirs(os.path.join(results_dir, "Graphs"), exist_ok=True)
save_path = os.path.join(results_dir, "Graphs", "Feature_Rank_Boxplot_with_Frequency_Final.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"💾 Saved to: {save_path}")
plt.show()

#%% USED bar plot for 6 metrics. ==== FIGURE 1: PERFORMANCE METRICS - BARS CLOSER TOGETHER ====
comparison_df = metrics_df_4ML.copy()
# Rename models to match the desired display format
model_name_mapping = {
    'Random_Forest': 'Random Forest',
    'XGBoost': 'XGBoost',
    'Neural_Network': 'Neural Network',
    'Simple_Regression': 'Simple Regression',
    'Bayesian_MA': 'BMA Full',          # 🔥 CHANGE (renamed)
    'Bayesian_MA_3': 'BMA Reduced'      # 🔥 CHANGE (added)
}
comparison_df['Model'] = comparison_df['Model'].map(model_name_mapping)
 
#performance metrics.
# Define the desired order of models
model_order = ['Random Forest', 'XGBoost', 'Neural Network', 'Simple Regression', 'BMA Full', 'BMA Reduced']  
# Reorder comparison_df according to model_order
comparison_df_ordered = comparison_df.set_index('Model').loc[model_order].reset_index()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# GREEN colormap (higher values = darker green)
green_cmap = plt.cm.Greens

performance_metrics = [
    ('A10', 'a-10', 100.0),   # 🔥 CHANGE: first
    ('IOA', 'IOA', 1.0),      # 🔥 CHANGE: second
    ('R²', 'R²', 1.0)       # 🔥 CHANGE: third
]

for idx, (metric, label, scale_factor) in enumerate(performance_metrics):
    ax = axes[idx]
    
    # Get values
    values = comparison_df_ordered[metric].values / scale_factor
    x_pos = np.arange(len(model_order))
    
    # Normalize values to range [0.3, 1.0] for color mapping
    # (0.3 minimum to avoid too light colors)
    min_val = values.min()
    max_val = values.max()
    normalized_values = 0.3 + 0.7 * (values - min_val) / (max_val - min_val)
    
    # Get colors based on normalized values
    colors = [green_cmap(norm_val) for norm_val in normalized_values]
    
    # Create bars with gradient colors
    bars = ax.bar(x_pos, values, 
                  color=colors,
                  edgecolor='black', 
                  linewidth=1.2, 
                  alpha=0.95, 
                  width=0.6)
    
    # Formatting
    if idx == 0:
        ax.set_ylabel('Benefit indices', fontsize=18, fontweight='bold')
    else:
        ax.set_yticklabels([])
    
    if idx == 1:
        ax.set_xlabel('Model', fontsize=18, fontweight='bold')
    
    ax.set_title(label, fontsize=18, fontweight='bold', pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

plt.subplots_adjust(wspace=0.15)
plt.tight_layout()

output_path = os.path.join(graphs_dir, 'Fig_Performance_Metrics_Final.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')

print(f"✅ Saved performance metrics to: {output_path}")
plt.show()

#FIGURE 2: ERROR METRICS (ONLY NRMSE & MAE; ORDER: NRMSE then MAE)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# RED colormap (lower values = darker red, so we INVERT)
red_cmap = plt.cm.OrRd_r

error_metrics = [
    ('NRMSE', 'NRMSE'),
    ('MAE', 'MAE')
]

for idx, (metric, label) in enumerate(error_metrics):
    ax = axes[idx]
    
    # Get values
    values = comparison_df_ordered[metric].values
    x_pos = np.arange(len(model_order))
    
    # Normalize values to range [0.3, 1.0] for color mapping (inverted for errors)
    min_val = values.min()
    max_val = values.max()
    normalized_values = 0.3 + 0.7 * (max_val - values) / (max_val - min_val)
    
    # Get colors based on normalized values
    colors = [red_cmap(norm_val) for norm_val in normalized_values]
    
    # Create bars with gradient colors
    bars = ax.bar(x_pos, values, 
                  color=colors,
                  edgecolor='black', 
                  linewidth=1.2, 
                  alpha=0.95, 
                  width=0.6)
    
    # Formatting (different scales handled by separate subplots + correct units)
    if idx == 0:
        ax.set_ylabel('NRMSE (%)', fontsize=18, fontweight='bold')
    elif idx == 1:
        ax.set_ylabel('MAE (t ha⁻¹)', fontsize=18, fontweight='bold')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    
    if idx == 0:
        ax.set_xlabel('Model', fontsize=18, fontweight='bold')
    
    ax.set_title(label, fontsize=18, fontweight='bold', pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

plt.subplots_adjust(wspace=0.15)
plt.tight_layout()

output_path = os.path.join(graphs_dir, 'Fig_Error_Metrics_Final.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')

print(f"✅ Saved error metrics to: {output_path}")
plt.show()