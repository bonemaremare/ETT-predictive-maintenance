import pandas as pd
import matplotlib.pyplot as plt


# 設定
DATA_PATH = 'ETTh1.csv'  # 検証対象のデータ
TARGET_COL = 'OT'        # ターゲット（温度）

df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
df = df.sort_index()

print(f"データセット概要 ({DATA_PATH})")
print(f"データ期間: {df.index.min()} 〜 {df.index.max()}")
print(f"総データ数: {len(df)} 時間分 (約 {len(df)/24/30:.1f} ヶ月)")
print(f"温度の範囲: 最小 {df[TARGET_COL].min():.1f}℃ 〜 最大 {df[TARGET_COL].max():.1f}℃\n")

# 2年分の温度推移のグラフ
plt.figure(figsize=(15, 5))
plt.plot(df.index, df[TARGET_COL], color='blue', linewidth=0.5)
plt.title(f'2-Year Temperature ({DATA_PATH}) ')
plt.xlabel('Date')
plt.ylabel('Oil Temperature (OT) [°C]')
plt.grid(True)
plt.tight_layout()
plt.show()
