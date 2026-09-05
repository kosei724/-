import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from dataloader import load_data
from features import make_windows, FEATURE_COLS

df = load_data("all", panda=True, includeNan=True, fillNan=True)
X, y, games = make_windows(df, window_size=5, label_threshold=0.5)

th = 0.05
all_fps = []

for test_game in sorted(games.unique()):
    train_mask = (games != test_game)
    test_mask = (games == test_game)

    model = RandomForestClassifier(random_state=0, class_weight="balanced", n_jobs=-1)
    model.fit(X[train_mask], y[train_mask])
    proba = model.predict_proba(X[test_mask])[:, 1]
    y_test = y[test_mask].values

    y_pred = (proba >= th).astype(int)

    # この試合の元データを取得
    df_game = df[df["game"] == test_game].reset_index(drop=True)

    # テストデータの窓の開始位置を再構築
    starts = list(range(0, len(df_game) - 5 + 1, 1))

    # 誤検出の窓を特定
    fp_indices = np.where(fp_mask := (y_pred == 1) & (y_test == 0))[0]

    for fi in fp_indices:
        start = starts[fi]
        window_labels = df_game.iloc[start:start + 5]["label"].tolist()
        all_fps.append({
            "game": test_game,
            "start": start,
            "labels": window_labels,
            "proba": proba[fi],
        })

print(f"誤検出(FP)の総数: {len(all_fps)}")

print(f"\n=== 誤検出の窓に含まれるラベル ===")
label_counts = {}
for fp in all_fps:
    for lab in fp["labels"]:
        lab = str(lab)
        label_counts[lab] = label_counts.get(lab, 0) + 1

for lab, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {lab:25s}: {cnt:4d}回")

print(f"\n=== 確率が高い誤検出 上位20件 ===")
all_fps.sort(key=lambda x: -x["proba"])
for i, fp in enumerate(all_fps[:20]):
    labels_str = ", ".join(str(l) for l in fp["labels"])
    print(f"  {i+1:2d}. [{fp['game']}] 行{fp['start']:5d} "
          f"確率{fp['proba']:.3f} ラベル: {labels_str}")