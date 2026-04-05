import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib, json
from feature_engineering import FEATURE_COLS, build_features

# Load and prepare
df = pd.read_csv(r'C:\Users\ASUS\Documents\skycity_project\data\SkyCity Auckland Restaurants & Bars.csv')
df = build_features(df)

X = df[FEATURE_COLS]
y = df['TotalNetProfit']

# Handle any NaN
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for linear models
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    mae  = mean_absolute_error(y_te, preds)
    cv   = cross_val_score(model, X_tr, y_tr, cv=5,
                            scoring='r2', n_jobs=-1).mean()
    print(f"{name:30s} RMSE={rmse:,.0f}  R²={r2:.4f}  MAE={mae:,.0f}  CV-R²={cv:.4f}")
    return {'rmse': rmse, 'r2': r2, 'mae': mae, 'cv_r2': cv}

results = {}
models  = {}

# Linear baseline (scaled)
lr = Ridge(alpha=1.0)
results['Ridge'] = evaluate('Ridge Regression', lr, X_train_sc, X_test_sc, y_train, y_test)
models['Ridge']  = lr

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                            min_samples_leaf=2, random_state=42, n_jobs=-1)
results['RandomForest'] = evaluate('Random Forest', rf, X_train, X_test, y_train, y_test)
models['RandomForest']  = rf

# Gradient Boosting
gbm = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                  max_depth=5, subsample=0.8, random_state=42)
results['GBM'] = evaluate('Gradient Boosting', gbm, X_train, X_test, y_train, y_test)
models['GBM']  = gbm

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                               max_depth=6, subsample=0.8, colsample_bytree=0.8,
                               random_state=42, n_jobs=-1, verbosity=0)
results['XGBoost'] = evaluate('XGBoost', xgb_model, X_train, X_test, y_train, y_test)
models['XGBoost']  = xgb_model

# Save all models
for name, model in models.items():
    joblib.dump(model, f'models/{name}.pkl')

# Save best model info
best = max(results, key=lambda k: results[k]['r2'])
print(f"\nBest model: {best} (R²={results[best]['r2']:.4f})")
with open('models/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save feature list for app
joblib.dump(FEATURE_COLS, 'models/feature_cols.pkl')
joblib.dump({'best_model': best, 'results': results}, 'models/summary.pkl')
print("All models saved to models/")