import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dataloader import load_data
from features import make_windows

df = load_data("all", panda=True, includeNan=True, fillNan=True)
X, y, games = make_windows(df, window_size=5, label_threshold=0.5)

train_mask = (games != "ricoh")
model = RandomForestClassifier(random_state=0, class_weight="balanced", n_jobs=-1)
model.fit(X[train_mask], y[train_mask])

# 学習に使ったデータ自体を予測
y_train = y[train_mask].values
train_proba = model.predict_proba(X[train_mask])[:, 1]
train_pred = (train_proba >= 0.5).astype(int)

print("=== 訓練データでの予測 ===")
print(f"訓練データの正例数: {y_train.sum()}")
print(f"TP: {((train_pred==1) & (y_train==1)).sum()}")
print(f"正例の予測確率: 平均{train_proba[y_train==1].mean():.4f} "
      f"最大{train_proba[y_train==1].max():.4f}")
print(f"負例の予測確率: 平均{train_proba[y_train==0].mean():.4f}")