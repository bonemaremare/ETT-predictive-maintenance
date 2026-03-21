import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, classification_report


# パラメータ・運用設定

DATA_PATH = 'ETTh2.csv'           # データ
TARGET_COL = 'OT'                 
HORIZON = 1                 # 予測する時間

# しきい値
ABSOLUTE_THRESHOLD = 85.0         # 絶対限界温度
DYNAMIC_SIGMA = 2.5               # 動的しきい値の乗数
DYNAMIC_WINDOW_HOURS = 24 * 7     # 平熱を計算する期間（過去168時間）


df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
df = df.sort_index()

# 動的しきい値の計算
df['rolling_mean'] = df[TARGET_COL].rolling(window=DYNAMIC_WINDOW_HOURS).mean()
df['rolling_std']  = df[TARGET_COL].rolling(window=DYNAMIC_WINDOW_HOURS).std()
df['dynamic_threshold'] = df['rolling_mean'] + (DYNAMIC_SIGMA * df['rolling_std'])

# ターゲット変数（horizon時間後）
df['target_OT'] = df[TARGET_COL].shift(-HORIZON)
df['target_threshold'] = df['dynamic_threshold'].shift(-HORIZON)

# 特徴量エンジニアリング
df['hour'] = df.index.hour
df['month'] = df.index.month
df['OT_lag_1'] = df[TARGET_COL].shift(1)
df['OT_lag_24'] = df[TARGET_COL].shift(24)
df['MULL_diff_1h'] = df['MULL'].diff(1) 
df = df.dropna()


# 秋までを学習し、冬の期間をテスト
train_df = df.loc[:'2017-10-31']
test_df  = df.loc['2017-11-01':'2018-01-31']

features = ['hour', 'month', 'OT_lag_1', 'OT_lag_24', 'MULL_diff_1h']
X_train, y_train = train_df[features], train_df['target_OT']
X_test,  y_test  = test_df[features],  test_df['target_OT']


#モデル学習
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
lgbm_preds = model.predict(X_test)

naive_mse = mean_squared_error(y_test, test_df[TARGET_COL])
lgbm_mse = mean_squared_error(y_test, lgbm_preds)

print(f"冬期テスト (Horizon: {HORIZON}h) ")
print(f"Error Reduction: {((naive_mse - lgbm_mse) / naive_mse * 100):.1f}% 削減\n")

actual_danger = (y_test >= test_df['target_threshold']) | (y_test >= ABSOLUTE_THRESHOLD)
predicted_danger = (lgbm_preds >= test_df['target_threshold']) | (lgbm_preds >= ABSOLUTE_THRESHOLD)

print(classification_report(actual_danger, predicted_danger, target_names=['Normal', 'Danger'], zero_division=0))
