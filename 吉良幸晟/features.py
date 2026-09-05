"""
スライディングウィンドウによる特徴量抽出
（本研究で新規に作成したファイル）
"""
import numpy as np
import pandas as pd

# 特徴量として使用する変数
FEATURE_COLS = [
    "Velocity", "Acceleration",
    # "Latitude", "Longitude",
    "Acceleration.forward", "Acceleration.side", "Acceleration.up",
    "Rotation.roll", "Rotation.pitch", "Rotation.yaw",
]


def make_windows(df, window_size=5, stride=1, label_threshold=0.5,
                 feature_cols=None):
    """
    試合ごとにスライディングウィンドウで特徴量を作成する

    Parameters
    ----------
    df : DataFrame
        dataloader.load_data("all") の出力
    window_size : int
        ウィンドウ幅（秒）
    stride : int
        窓をずらす幅（秒）
    label_threshold : float
        窓内のタックル割合がこの値以上ならタックルとする
    feature_cols : list or None
        使用する変数。Noneの場合はFEATURE_COLSを使用

    Returns
    -------
    X : DataFrame  特徴量行列
    y : Series     ラベル（0 or 1）
    games : Series 試合名
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_list, y_list, game_list = [], [], []

    # 試合ごとに処理（境界をまたがないため）
    for game_name in df["game"].unique():
        df_game = df[df["game"] == game_name].reset_index(drop=True)

        n_rows = len(df_game)
        for start in range(0, n_rows - window_size + 1, stride):
            window = df_game.iloc[start:start + window_size]

            # 欠損があればスキップ
            if window[feature_cols].isna().any().any():
                continue

            feat = {}

            # ===== 基本統計量 + 時系列特徴量 =====
            for col in feature_cols:
                values = window[col].values
                n = len(values)

                # 窓全体の統計量
                feat[f"{col}_max"] = np.max(values)
                feat[f"{col}_min"] = np.min(values)
                feat[f"{col}_mean"] = np.mean(values)
                feat[f"{col}_std"] = np.std(values)
                feat[f"{col}_range"] = np.max(values) - np.min(values)

                # 時間的な変化を捉える特徴量
                first = values[:n // 2]
                last = values[-(n // 2):]
                feat[f"{col}_first_mean"] = np.mean(first)
                feat[f"{col}_last_mean"] = np.mean(last)
                feat[f"{col}_diff"] = np.mean(last) - np.mean(first)
                feat[f"{col}_start_end"] = values[-1] - values[0]
                feat[f"{col}_argmax"] = int(np.argmax(values)) / n
                feat[f"{col}_argmin"] = int(np.argmin(values)) / n

            # ===== 変数間の組み合わせ特徴量 =====
            v = window[feature_cols].values
            col_idx = {c: i for i, c in enumerate(feature_cols)}

            if "Velocity" in col_idx and "Rotation.roll" in col_idx:
                vel = v[:, col_idx["Velocity"]]
                roll = v[:, col_idx["Rotation.roll"]]
                pitch = v[:, col_idx["Rotation.pitch"]]
                acc = v[:, col_idx["Acceleration"]]

                vel_drop = vel[0] - vel[-1]
                feat["vel_drop"] = vel_drop
                feat["vel_drop_x_roll_std"] = vel_drop * np.std(roll)
                feat["vel_drop_x_pitch_std"] = vel_drop * np.std(pitch)
                feat["acc_range_x_roll_range"] = ((np.max(acc) - np.min(acc))
                                                  * (np.max(roll) - np.min(roll)))
                feat["acc_range_x_pitch_range"] = ((np.max(acc) - np.min(acc))
                                                   * (np.max(pitch) - np.min(pitch)))
                vel_change = np.mean(vel[:len(vel) // 2]) - np.mean(vel[-(len(vel) // 2):])
                feat["vel_change_x_acc_std"] = vel_change * np.std(acc)
                feat["roll_std_x_pitch_std"] = np.std(roll) * np.std(pitch)

            # ラベルの決定
            tackle_ratio = window["is_tackle"].mean()
            label = 1 if tackle_ratio >= label_threshold else 0

            X_list.append(feat)
            y_list.append(label)
            game_list.append(game_name)

    X = pd.DataFrame(X_list)
    y = pd.Series(y_list, name="label")
    games = pd.Series(game_list, name="game")

    return X, y, games