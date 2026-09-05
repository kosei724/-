import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from dataloader import load_data
from features import make_windows

df = load_data("all", panda=True, includeNan=True, fillNan=True)
X, y, games = make_windows(df, window_size=5, label_threshold=0.5)

# ricohをテストにする
train_mask = (games != "ricoh")
test_mask = (games == "ricoh")

model = RandomForestClassifier(random_state=0, class_weight="balanced", n_jobs=-1)
model.fit(X[train_mask], y[train_mask])

proba = model.predict_proba(X[test_mask])[:, 1]  # タックルである確率
y_test = y[test_mask].values

print("=== 予測確率の分布 ===")
print(f"正例(タックル)の確率: 平均{proba[y_test==1].mean():.4f} "
      f"最大{proba[y_test==1].max():.4f}")
print(f"負例の確率          : 平均{proba[y_test==0].mean():.4f} "
      f"最大{proba[y_test==0].max():.4f}")
print()
print("正例の確率 上位10件:", np.sort(proba[y_test==1])[::-1][:10].round(3))
print()

# 閾値を下げたらどうなるか
print("=== 閾値を変えた場合 ===")
for th in [0.5, 0.3, 0.2, 0.1, 0.05]:
    pred = (proba >= th).astype(int)
    tp = ((pred==1) & (y_test==1)).sum()
    fp = ((pred==1) & (y_test==0)).sum()
    fn = ((pred==0) & (y_test==1)).sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f"閾値{th}: TP={tp:3d} FP={fp:4d} FN={fn:3d} "
          f"適合率={prec:.3f} 再現率={rec:.3f} F1={f1:.3f}")