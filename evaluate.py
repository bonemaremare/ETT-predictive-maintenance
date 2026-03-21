import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 設定
DATA_PATH = 'ETTh2.csv'
TARGET_COL = 'OT'
HORIZON = 6  



df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
df = df.sort_index()

# たーゲット
df['target_OT'] = df[TARGET_COL].shift(-HORIZON)

# 特徴量エンジニアリング
df['hour'] = df.index.hour
df['month'] = df.index.month
df['OT_lag_1'] = df[TARGET_COL].shift(1)
df['OT_lag_24'] = df[TARGET_COL].shift(24)
df['MULL_diff_1h'] = df['MULL'].diff(1)

df = df.dropna()

#期間
train_df = df.loc[:'2017-05-31']
test_df  = df.loc['2017-06-01':'2017-08-31']

features = ['hour', 'month', 'OT_lag_1', 'OT_lag_24', 'MULL_diff_1h']
X_train, y_train = train_df[features], train_df['target_OT']
X_test,  y_test  = test_df[features],  test_df['target_OT']


# LightGBMによる回帰モデル
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
lgbm_preds = model.predict(X_test)

# Naiveベースライン（horizon時間後に同じ温度を予測）
naive_preds = test_df[TARGET_COL]



mae_model = mean_absolute_error(y_test, lgbm_preds)
mae_naive = mean_absolute_error(y_test, naive_preds)

mse_model = mean_squared_error(y_test, lgbm_preds)
mse_naive = mean_squared_error(y_test, naive_preds)

r2_model = r2_score(y_test, lgbm_preds)


print(f"========== Regression Evaluation (Horizon: {HORIZON}h) ==========")
print(f"[MAE] Model: {mae_model:.2f} degC | Naive: {mae_naive:.2f} degC")
print(f"[MSE] Model: {mse_model:.2f} | Naive: {mse_naive:.2f}")
print(f"[R2 Score] Model: {r2_model:.3f}\n")


plt.figure(figsize=(15, 6))

# 最初の500時間を描写
plt.plot(y_test.index[:500], y_test.values[:500], label='Actual Temperature', color='blue', linewidth=2)
plt.plot(y_test.index[:500], lgbm_preds[:500], label='AI Prediction', color='orange', linestyle='dashed', linewidth=2)
plt.plot(y_test.index[:500], naive_preds.values[:500], label='Naive Baseline', color='green', alpha=0.4, linestyle='dotted')


plt.title(f'Temperature Prediction (Horizon: {HORIZON}h) - First 500 Hours')
plt.ylabel('Oil Temperature (OT) [degC]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
