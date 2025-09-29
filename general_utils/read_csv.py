import pandas as pd

csv_path = r"C:\Users\User\Desktop\data\sample 2\2025-09-29_full_detections.csv"
df = pd.read_csv(csv_path)
print(df.columns.tolist())
print(set(df['category']))
print(f"X: min={df['x'].min()}, max={df['x'].max()}")
print(f"Y: min={df['y'].min()}, max={df['y'].max()}")

df = df[df['category'] == 'Category 2']
print(len(df))