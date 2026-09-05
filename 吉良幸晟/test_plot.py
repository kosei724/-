import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataloader import load_data

df = load_data("all", panda=True, includeNan=True, fillNan=True)

# ricohのタックル区間を特定
g = df[df["game"] == "ricoh"].reset_index(drop=True)
idx = g[g["is_tackle"] == 1].index.tolist()

# 連続区間に分ける
events = []
start = idx[0]
prev = idx[0]
for i in idx[1:]:
    if i - prev > 1:
        events.append((start, prev))
        start = i
    prev = i
events.append((start, prev))

print(f"タックルイベント数: {len(events)}")
for i, (s, e) in enumerate(events):
    print(f"  {i+1}: 行{s}〜{e} ({e-s+1}秒)")

# 最初の6イベントを可視化
fig, axes = plt.subplots(6, 3, figsize=(15, 18))
for i, (s, e) in enumerate(events[:6]):
    lo, hi = max(0, s-8), min(len(g), e+9)
    seg = g.iloc[lo:hi]
    x = range(lo, hi)

    for j, col in enumerate(["Velocity", "Acceleration", "Rotation.roll"]):
        ax = axes[i][j]
        ax.plot(x, seg[col].values, lw=1.5)
        ax.axvspan(s, e, alpha=0.25, color="red")
        ax.set_title(f"Event{i+1} {col}", fontsize=9)
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("tackle_check.png", dpi=100)
print("\ntackle_check.png を保存しました")