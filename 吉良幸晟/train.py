"""
タックル検出の学習・評価
Leave-One-Game-Out交差検証によるベースライン実験

（本研究で新規に作成したファイル）
結果保存の形式は先行研究のLOGOCV.pyを参考にした
"""
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

from dataloader import load_data
from features import make_windows


# ===== 実験設定 =====
CONFIG = {
    "window_size": 5,
    "stride": 1,
    "label_threshold": 0.5,
    "model": "RandomForestClassifier",
    "class_weight": "balanced",  
    "n_seeds": 10,
}


def run_experiment(X, y, games, config):
    """Leave-One-Game-Out交差検証を実行する"""
    game_list = sorted(games.unique())
    n_seeds = config["n_seeds"]

    # 結果を貯める辞書 {試合名: {指標: [シード数分の値]}}
    results = {g: {"accuracy": [], "precision": [], "recall": [], "f1": [],
                   "TP": [], "FP": [], "FN": [], "TN": []} for g in game_list}
    importances = []

    for seed in range(n_seeds):
        print(f"シード {seed} ...", end=" ", flush=True)

        for test_game in game_list:
            train_mask = (games != test_game)
            test_mask = (games == test_game)

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            model = RandomForestClassifier(
                random_state=seed,
                class_weight=config["class_weight"],
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            r = results[test_game]
            r["accuracy"].append(accuracy_score(y_test, y_pred))
            r["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            r["recall"].append(recall_score(y_test, y_pred, zero_division=0))
            r["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            r["TP"].append(int(tp))
            r["FP"].append(int(fp))
            r["FN"].append(int(fn))
            r["TN"].append(int(tn))

            importances.append(model.feature_importances_)

        print("完了", flush=True)

    return results, np.array(importances)


def print_results(results, X, importances):
    """結果を表示する"""
    game_list = sorted(results.keys())

    print("\n" + "=" * 60)
    print("試合ごとの結果（平均 ± 標準偏差）")
    print("=" * 60)

    for g in game_list:
        r = results[g]
        print(f"\n[{g}]")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            vals = r[metric]
            print(f"  {metric:10s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        print(f"  混同行列    : TP={np.mean(r['TP']):.1f} "
              f"FP={np.mean(r['FP']):.1f} "
              f"FN={np.mean(r['FN']):.1f} "
              f"TN={np.mean(r['TN']):.1f}")

    print("\n" + "=" * 60)
    print("3試合平均")
    print("=" * 60)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        all_vals = [v for g in game_list for v in results[g][metric]]
        print(f"  {metric:10s}: {np.mean(all_vals):.3f} ± {np.std(all_vals):.3f}")

    print("\n" + "=" * 60)
    print("特徴量重要度 上位20")
    print("=" * 60)
    mean_imp = importances.mean(axis=0)
    imp_series = pd.Series(mean_imp, index=X.columns).sort_values(ascending=False)
    for i, (name, val) in enumerate(imp_series.head(20).items(), 1):
        print(f"  {i:2d}. {name:30s} {val:.4f}")

    return imp_series


def save_results(config, results, imp_series):
    """結果をJSONに保存する（先行研究のLOGOCV.pyを参考）"""
    game_list = sorted(results.keys())

    summary = {}
    for g in game_list:
        r = results[g]
        summary[g] = {
            metric: {"mean": float(np.mean(r[metric])),
                     "std": float(np.std(r[metric]))}
            for metric in ["accuracy", "precision", "recall", "f1"]
        }
        summary[g]["confusion_matrix"] = {
            k: float(np.mean(r[k])) for k in ["TP", "FP", "FN", "TN"]
        }

    overall = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        all_vals = [v for g in game_list for v in results[g][metric]]
        overall[metric] = {"mean": float(np.mean(all_vals)),
                           "std": float(np.std(all_vals))}

    output = {
        "config": config,
        "per_game": summary,
        "overall": overall,
        "feature_importance_top20": {
            name: float(val) for name, val in imp_series.head(20).items()
        },
        "raw": {g: results[g] for g in game_list},
    }

    filename = "./結果/result_" + datetime.datetime.now().strftime("%y%m%d%H%M%S") + ".json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果を保存しました: {filename}")


def main():
    print("設定:", CONFIG)
    print("\nデータ読み込み中...")
    df = load_data("all", panda=True, includeNan=True, fillNan=True)

    print("特徴量作成中...")
    X, y, games = make_windows(
        df,
        window_size=CONFIG["window_size"],
        stride=CONFIG["stride"],
        label_threshold=CONFIG["label_threshold"],
    )
    print(f"  特徴量: {X.shape}  正例: {int(y.sum())}個")

    print("\n学習・評価中...")
    results, importances = run_experiment(X, y, games, CONFIG)

    imp_series = print_results(results, X, importances)
    save_results(CONFIG, results, imp_series)


if __name__ == "__main__":
    main()