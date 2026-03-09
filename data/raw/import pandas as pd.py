
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────
df = pd.read_csv('DP_LIVE_02012022050459635.csv')
df.columns = df.columns.str.strip().str.replace('\ufeff', '')
df = df[['LOCATION', 'TIME', 'Value']].dropna(subset=['Value'])
df.columns = ['country', 'year', 'production']
df['year'] = df['year'].astype(int)
df = df.sort_values(['country', 'year']).reset_index(drop=True)

print(f"Shape: {df.shape}")
print(f"Countries: {df['country'].nunique()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING (no data leakage)
#    All features are derived from past values only
# ─────────────────────────────────────────────
df['production_lag1'] = df.groupby('country')['production'].shift(1)
df['production_lag2'] = df.groupby('country')['production'].shift(2)
df['production_lag3'] = df.groupby('country')['production'].shift(3)

# Rolling average of the 3 years BEFORE the current year
df['rolling_avg_3'] = df.groupby('country')['production'].transform(
    lambda x: x.shift(1).rolling(3).mean()
)

# Encode country as numeric
df['country_code'] = df['country'].astype('category').cat.codes

# Drop rows with NaNs introduced by lagging
df = df.dropna()

# ─────────────────────────────────────────────
# 3. TIME-BASED TRAIN/TEST SPLIT
#    Train on years <= 2013, test on 2014-2017
#    (avoids future data leaking into training)
# ─────────────────────────────────────────────
SPLIT_YEAR = 2013
features = ['year', 'country_code', 'production_lag1',
            'production_lag2', 'production_lag3', 'rolling_avg_3']

X = df[features]
y = df['production']

X_train = X[df['year'] <= SPLIT_YEAR]
y_train = y[df['year'] <= SPLIT_YEAR]
X_test  = X[df['year'] > SPLIT_YEAR]
y_test  = y[df['year'] > SPLIT_YEAR]

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Test years: {sorted(df[df['year'] > SPLIT_YEAR]['year'].unique())}\n")

# ─────────────────────────────────────────────
# 4. TRAIN MODELS & EVALUATE R²
# ─────────────────────────────────────────────
models = {
    'Linear Regression' : LinearRegression(),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("─── With lag features (realistic setup) ───")
for name, model in models.items():
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"  {name:<25} R² = {r2:.4f}")

# ─────────────────────────────────────────────
# 5. ABLATION — remove lag features
#    Shows what each model contributes on its own
# ─────────────────────────────────────────────
X2       = df[['year', 'country_code']]
X2_train = X2[df['year'] <= SPLIT_YEAR]
X2_test  = X2[df['year'] > SPLIT_YEAR]

print("\n─── Without lag features (year + country only) ───")
for name, model in models.items():
    model.fit(X2_train, y_train)
    r2 = r2_score(y_test, model.predict(X2_test))
    print(f"  {name:<25} R² = {r2:.4f}")

# ─────────────────────────────────────────────
# 6. KEY DIAGNOSTIC
# ─────────────────────────────────────────────
corr = df['production_lag1'].corr(df['production'])
print(f"\nCorrelation between lag1 and production: {corr:.4f}")
print("(Explains why high R² is legitimate — oil output is highly autocorrelated)")