import pandas as pd

DATA_PATH = 'ETTh2.csv'

df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')

print(" 特徴量間の相関行列（相関係数）")

features_to_check = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
correlation_matrix = df[features_to_check].corr()


print(correlation_matrix.round(3))
