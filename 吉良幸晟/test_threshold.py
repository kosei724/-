import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from dataloader import load_data
from features import make_windows

df = load_data("all", panda=True, includeNan=True, fillNan=True)
X, y, games = make_windows(df, window_size=5, label_threshold=0.5)

print("閾値を変えた場合のF1値（10シード平均）\n")

best_overall = {"th": 0, "f1": 0}

for th in [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]:
    f1s, precs, recs = [], [], []
    
    for seed in range(10):
        f1_per_game, prec_per_game, rec_per_game = [], [], []
        
        for test_game in sorted(games.unique()):
            train_mask = (games != test_game)
            test_mask = (games == test_game)
            
            model = RandomForestClassifier(random_state=seed, 
                                           class_weight="balanced", n_jobs=-1)
            model.fit(X[train_mask], y[train_mask])
            proba = model.predict_proba(X[test_mask])[:, 1]
            y_test = y[test_mask].values
            
            pred = (proba >= th).astype(int)
            f1_per_game.append(f1_score(y_test, pred, zero_division=0))
            prec_per_game.append(precision_score(y_test, pred, zero_division=0))
            rec_per_game.append(recall_score(y_test, pred, zero_division=0))
        
        f1s.append(np.mean(f1_per_game))
        precs.append(np.mean(prec_per_game))
        recs.append(np.mean(rec_per_game))
    
    mean_f1 = np.mean(f1s)
    print(f"閾値{th:.3f}: F1={mean_f1:.3f} "
          f"適合率={np.mean(precs):.3f} 再現率={np.mean(recs):.3f}")
    
    if mean_f1 > best_overall["f1"]:
        best_overall = {"th": th, "f1": mean_f1}

print(f"\n★ 最良: 閾値{best_overall['th']:.3f} → F1={best_overall['f1']:.3f}")