import pandas as pd
from dataloader import load_data
from features import make_windows

df = load_data("all", panda=True, includeNan=True, fillNan=True)
X, y, games = make_windows(df, window_size=5, label_threshold=0.5)

cols = ["Acceleration_min", "Acceleration_std", "Acceleration_range",
        "Acceleration.side_range", "Acceleration.up_range",
        "Rotation.roll_std", "Rotation.pitch_std", "Velocity_max"]

print("=== 全窓の平均（試合間でスケールが違うか）===")
rows = []
for g in sorted(games.unique()):
    mask = (games == g)
    rows.append(X.loc[mask, cols].mean())
comp = pd.DataFrame(rows, index=sorted(games.unique()))
print(comp.round(3).T)

print("\n=== 正例だけの平均（タックルの特徴が試合で違うか）===")
rows = []
for g in sorted(games.unique()):
    mask = (games == g) & (y == 1)
    rows.append(X.loc[mask, cols].mean())
comp_pos = pd.DataFrame(rows, index=sorted(games.unique()))
print(comp_pos.round(3).T)

print("\n=== 正例と負例の差（試合ごと）===")
for g in sorted(games.unique()):
    pos = X.loc[(games == g) & (y == 1), cols].mean()
    neg = X.loc[(games == g) & (y == 0), cols].mean()
    print(f"\n[{g}] 正例/負例の比")
    print((pos / neg).round(2))