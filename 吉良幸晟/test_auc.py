import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from dataloader import load_data
from features import make_windows

df = load_data("all", panda=True, includeNan=True, fillNan=True)
X, y, games = make_windows(df, window_size=5, label_threshold=0.5)

for test_game in sorted(games.unique()):
    train_mask = (games != test_game)
    test_mask = (games == test_game)

    model = RandomForestClassifier(random_state=0, class_weight="balanced", n_jobs=-1)
    model.fit(X[train_mask], y[train_mask])
    proba = model.predict_proba(X[test_mask])[:, 1]
    y_test = y[test_mask].values

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    # 確率の高い順に並べて、上位K件に正例が何個入るか
    order = np.argsort(proba)[::-1]
    n_pos = int(y_test.sum())
    print(f"\n[{test_game}] 正例{n_pos}個 / 全{len(y_test)}窓")
    print(f"  ROC-AUC     : {auc:.3f}")
    print(f"  PR-AUC      : {ap:.4f}")
    for k in [50, 100, 200, 500]:
        hit = y_test[order[:k]].sum()
        print(f"  上位{k:4d}件中の正例: {hit:3d}個 (適合率{hit/k:.3f} 再現率{hit/n_pos:.3f})")