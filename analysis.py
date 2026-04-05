import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from feature_engineering import FEATURE_COLS, build_features

df = pd.read_csv(r'C:\Users\ASUS\Documents\skycity_project\data\SkyCity Auckland Restaurants & Bars.csv')
df = build_features(df)
X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
y = df['TotalNetProfit']

# Load best model (change to your best)
model = joblib.load('models/XGBoost.pkl')

# SHAP feature importance
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS,
                   plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('models/shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Sensitivity analysis ---
def sensitivity_analysis(model, X_base, feature, values, feature_cols):
    profits = []
    for v in values:
        X_mod = X_base.copy()
        X_mod[feature] = v
        profits.append(model.predict(X_mod).mean())
    return profits

X_base = X.copy()
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

tests = [
    ('CommissionRate',      np.linspace(0.15, 0.45, 30)),
    ('InStoreShare',        np.linspace(0.0, 0.8, 30)),
    ('UE_share',            np.linspace(0.0, 0.8, 30)),
    ('SD_share',            np.linspace(0.0, 0.8, 30)),
    ('DeliveryCostPerOrder',np.linspace(0.5, 6.0, 30)),
    ('GrowthFactor',        np.linspace(0.95, 1.1, 30)),
]
for ax, (feat, vals) in zip(axes.flatten(), tests):
    profits = sensitivity_analysis(model, X_base, feat, vals, FEATURE_COLS)
    ax.plot(vals, profits, color='steelblue', lw=2)
    ax.axhline(y=model.predict(X_base).mean(), color='red',
                linestyle='--', alpha=0.6, label='Baseline')
    ax.set_title(f'Sensitivity: {feat}')
    ax.set_xlabel(feat)
    ax.set_ylabel('Avg Predicted Profit ($)')
    ax.legend(fontsize=8)

plt.suptitle('Profit Sensitivity Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('models/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
print("Analysis saved.")