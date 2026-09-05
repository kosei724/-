import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dataloader import load_data
from features import make_windows


def completion(predictions, max_gap=3):
    """
    補完処理（先行研究のutils.pyのロジックを参考）
    
    検出の隙間がmax_gap秒以内なら、その間を埋める
    先行研究ではスクラム用に15秒だったが、
    タックルは3〜7秒なので短くする
    """
    result = predictions.copy()
    i = 0
    while i < len(result):
        if result[i] == 1:
            # 次の検出を探す
            j = i + 1
            while j < len(result) and result[j] == 0:
                j += 1
            # 隙間がmax_gap以内なら埋める
            if j < len(result) and result[j] == 1 and (j - i - 1) <= max_gap:
                result[i:j] = 1
            i = j
        else:
            i += 1
    return result


def grouping(predictions, min_duration=2):
    """
    結合処理（先行研究のutils.pyのロジックを参考）
    
    連続する検出を1イベントにまとめる
    min_duration秒未満のイベントはノイズとして除去
    """
    events = []
    in_event = False
    start = 0
    
    for i, p in enumerate(predictions):
        if p == 1 and not in_event:
            start = i
            in_event = True
        elif p == 0 and in_event:
            if i - start >= min_duration:
                events.append((start, i - 1))
            in_event = False
    
    if in_event and len(predictions) - start >= min_duration:
        events.append((start, len(predictions) - 1))
    
    return events


def evaluation(true_events, pred_events, tolerance=5):
    """
    イベント単位の評価（先行研究のutils.pyのロジックを参考）
    
    真のイベントと予測イベントの時刻差がtolerance以内ならTP
    先行研究ではスクラム用に15秒だったが、タックル用に5秒に設定
    """
    matched_true = set()
    matched_pred = set()
    
    for pi, (ps, pe) in enumerate(pred_events):
        for ti, (ts, te) in enumerate(true_events):
            if ti in matched_true:
                continue
            if abs(ps - ts) <= tolerance or abs(pe - te) <= tolerance:
                matched_true.add(ti)
                matched_pred.add(pi)
                break
    
    tp = len(matched_pred)
    fp = len(pred_events) - tp
    fn = len(true_events) - len(matched_true)
    
    return tp, fp, fn


# データ準備
df = load_data("all", panda=True, includeNan=True, fillNan=True)
X, y, games = make_windows(df, window_size=5, label_threshold=0.3)

print("=" * 70)
print("フレーム単位 vs 補完・結合後のイベント単位")
print("補完: 先行研究のロジックを参考（秒数はタックル向けに調整）")
print("=" * 70)

for th in [0.01, 0.02, 0.03, 0.05]:
    for max_gap in [3, 5]:
        all_tp, all_fp, all_fn = 0, 0, 0
        all_true_ev, all_pred_ev = 0, 0
        
        for test_game in sorted(games.unique()):
            train_mask = (games != test_game)
            test_mask = (games == test_game)
            
            model = RandomForestClassifier(random_state=0,
                                           class_weight="balanced", n_jobs=-1)
            model.fit(X[train_mask], y[train_mask])
            proba = model.predict_proba(X[test_mask])[:, 1]
            y_test = y[test_mask].values
            
            y_pred = (proba >= th).astype(int)
            
            # 補完→結合→評価
            y_pred_comp = completion(y_pred, max_gap=max_gap)
            pred_events = grouping(y_pred_comp, min_duration=2)
            true_events = grouping(y_test, min_duration=1)
            tp, fp, fn = evaluation(true_events, pred_events, tolerance=5)
            
            all_tp += tp
            all_fp += fp
            all_fn += fn
            all_true_ev += len(true_events)
            all_pred_ev += len(pred_events)
        
        total_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        total_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        total_f1 = (2 * total_prec * total_rec / (total_prec + total_rec)
                    if (total_prec + total_rec) > 0 else 0)
        
        print(f"\n閾値{th} 補完{max_gap}秒: "
              f"真{all_true_ev}回 予測{all_pred_ev}回 → "
              f"TP={all_tp} FP={all_fp} FN={all_fn} | "
              f"適合率={total_prec:.3f} 再現率={total_rec:.3f} F1={total_f1:.3f}")