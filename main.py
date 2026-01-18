import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. DATA LOADING
def load_gas_turbine_data():
    files = ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv", "gt_2014.csv", "gt_2015.csv"]
    data_list = [pd.read_csv(f) for f in files]
    # Keep track of indices for temporal splitting later
    return data_list

data_years = load_gas_turbine_data()
df_all = pd.concat(data_years, axis=0)

# 2. EXPLORATORY DATA ANALYSIS (EDA)
plt.figure(figsize=(12, 8))
sns.heatmap(df_all.corr(), annot=True, cmap='RdBu', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

# 3. TEMPORAL SPLITTING (As per reference paper)
# Train: 2011-12 | Val: 2013 | Test: 2014-15
df_train = pd.concat(data_years[0:2])
df_val = data_years[2]
df_test = pd.concat(data_years[3:5])

features = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']
target_co = 'CO'
target_nox = 'NOX'

def get_xy(df):
    return df[features], df[target_co], df[target_nox]

X_train, y_co_train, y_nox_train = get_xy(df_train)
X_val, y_co_val, y_nox_val = get_xy(df_val)
X_test, y_co_test, y_nox_test = get_xy(df_test)

# 4. PREPROCESSING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. MODELING & EVALUATION FUNCTION
def evaluate_model(model, X_t, y_t, X_v, y_v, name="Model"):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    mae = mean_absolute_error(y_v, preds)
    r2 = r2_score(y_v, preds)
    print(f"{name} -> Val MAE: {mae:.4f} | Val R2: {r2:.4f}")
    return model

print("\n--- CO (Carbon Monoxide) Models ---")
rf_co = evaluate_model(RandomForestRegressor(n_estimators=100, random_state=42), 
                       X_train_scaled, y_co_train, X_val_scaled, y_co_val, "Random Forest")

mlp_co = evaluate_model(MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42), 
                        X_train_scaled, y_co_train, X_val_scaled, y_co_val, "MLP Neural Net")

ridge_co = evaluate_model(Ridge(alpha=1.0), 
                         X_train_scaled, y_co_train, X_val_scaled, y_co_val, "Ridge Regression")

print("\n--- NOx (Nitrogen Oxides) Models ---")
rf_nox = evaluate_model(RandomForestRegressor(n_estimators=100, random_state=42), 
                        X_train_scaled, y_nox_train, X_val_scaled, y_nox_val, "Random Forest")

mlp_nox = evaluate_model(MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42), 
                         X_train_scaled, y_nox_train, X_val_scaled, y_nox_val, "MLP Neural Net")

# 6. FINAL TEST (Example using RF on Test Set)
test_preds_nox = rf_nox.predict(X_test_scaled)
print(f"\nFinal Test R2 (NOx): {r2_score(y_nox_test, test_preds_nox):.4f}")
