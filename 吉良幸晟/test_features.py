import pandas as pd
from dataloader import load_data
from features import make_windows

df = load_data("all", panda=True, includeNan=True, fillNan=True)
X, y, games = make_windows(df, window_size=5, label_threshold=0.5)

print("特徴量の形状:", X.shape)
print()
print("試合ごとの窓数と正例数:")
for g in games.unique():
    mask = (games == g)
    print(f"  {g}: {mask.sum()}窓 / 正例{int(y[mask].sum())}個")
print()
print("正例と負例の平均比較（差が大きい上位10変数）:")
comp = pd.DataFrame({"正例": X[y==1].mean(), "負例": X[y==0].mean()})
comp["比"] = (comp["正例"] / comp["負例"]).abs()
print(comp.sort_values("比", ascending=False).head(10).round(3))