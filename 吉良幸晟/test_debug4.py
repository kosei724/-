import numpy as np
import pandas as pd
from dataloader import load_data
from features import make_windows

df = load_data("all", panda=True, includeNan=True, fillNan=True)
X, y, games = make_windows(df, window_size=5, label_threshold=0.5)

cols = ["Acceleration_min", "Rotation.roll_std", "Acceleration_range",
        "Rotation.pitch_std", "Velocity_max"]

print("=== 正例と負例の分布の重なり ===")
for c in cols:
    pos = X.loc[y==1, c]
    neg = X.loc[y==0, c]
    
    # 正例の代表値（絶対値が大きい方が特徴的なので符号に注意）
    if pos.mean() < 0:
        threshold = pos.quantile(0.5)      # 正例の中央値
        n_neg_beyond = (neg <= threshold).sum()   # それより極端な負例
    else:
        threshold = pos.quantile(0.5)
        n_neg_beyond = (neg >= threshold).sum()
    
    print(f"\n{c}")
    print(f"  正例の中央値: {threshold:.3f}")
    print(f"  これより極端な負例: {n_neg_beyond}個 (正例97個に対して {n_neg_beyond/97:.1f}倍)")

print("\n=== 複数条件を組み合わせた場合 ===")
# 正例らしい条件をすべて満たす窓を数える
cond = (
    (X["Acceleration_min"] <= X.loc[y==1, "Acceleration_min"].quantile(0.5)) &
    (X["Rotation.roll_std"] >= X.loc[y==1, "Rotation.roll_std"].quantile(0.5)) &
    (X["Acceleration_range"] >= X.loc[y==1, "Acceleration_range"].quantile(0.5))
)
print(f"3条件すべて満たす窓: {cond.sum()}個")
print(f"  うち正例: {(cond & (y==1)).sum()}個")
print(f"  うち負例: {(cond & (y==0)).sum()}個")