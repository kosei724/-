"""
データを読み込むデータローダーのファイル
（先行研究：飯田さんのdataloader.pyを改造）

改造内容（2026/08/26）：
- Variance関連の処理を削除（スクラム専用のため）
- 対象試合を3試合に限定
- load_datetime()を削除（デモ用のため不使用）
- フォルダ名をGPSファイルに修正
"""
import numpy as np
import pandas as pd
import os
import pickle
import pathlib

dataset_path = pathlib.Path('./試合データセット')
gnss_dir = 'GPSファイル'
acceleration_dir = '加速度ファイル'

# 使用する3試合のファイル名
FILES = {
    "ricoh":          ("gnss_タグ付け_Ricoh.csv",
                       "accelerometer_タグ付け_Ricoh.csv"),
    "toshiba_7OSUGI": ("gnss_タグ付け_Toshiba_7OSUGI.csv",
                       "accelerometer_タグ付け_Toshiba_7OSUGI.csv"),
    "toshiba":        ("gnss_タグ付け_Toshiba.csv",
                       "accelerometer_タグ付け_Toshiba.csv"),
}


def normalize_label(label):
    """ラベルの表記揺れを正規化する"""
    if pd.isna(label):
        return "Nan"
    label = str(label).strip().lower()          # Maul → maul
    label = label.replace("suport", "support")  # スペルミス対策
    return label


def is_tackle(label):
    """タックルなら1、それ以外は0を返す"""
    return 1 if "tackle" in str(label) else 0



def get_df(data, includeNan=False, fillNan=False):
    columns = ["label", "Velocity", "Acceleration", "Latitude", "Longitude",
               "imuOrientation.forward", "imuOrientation.side", "Facing",
               "Acceleration.forward", "Acceleration.side", "Acceleration.up",
               "imuAcceleration.forward", "imuAcceleration.side", "imuAcceleration.up",
               "Rotation.roll", "Rotation.pitch", "Rotation.yaw",  "game"]

    df = pd.DataFrame(data, columns=columns)

    if not includeNan:
        df2 = df.dropna(how='any').dropna(how='any', axis=1)
    else:
        df2 = df

    if fillNan:
        df3 = df2.fillna({"label": "Nan"})
    else:
        df3 = df2

    df3 = df3.copy()
    df3["is_tackle"] = df3["label"].apply(is_tackle)   # ← 追加

    return df3


def get_raw(data):
    x = []
    label = []
    for i in range(len(data)):
        x.append(data[i][1:])
        label.append(data[i][0])
    return x, label


def load_data(name, panda=True, includeNan=False, fillNan=False, interval=1, skip=[]):
    if name == "all":
        return load_all(panda, includeNan, fillNan, interval, skip)

    pickle_file = f"./前処理済みデータ/{name}_interval_{interval}.pickle"

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        if panda:
            return get_df(data, includeNan=includeNan, fillNan=fillNan)
        else:
            return get_raw(data)

    # ファイル名を取得
    gnss_file, acceleration_file = FILES[name]

    # GNSSデータ読み込み
    df_gnss = pd.read_csv(dataset_path / gnss_dir / gnss_file, low_memory=False)
    df_gnss = df_gnss.drop(range(10))

    # 加速度データ読み込み
    df_accelerometer = pd.read_csv(dataset_path / acceleration_dir / acceleration_file,
                                   low_memory=False)
    df_accelerometer = df_accelerometer.drop(range(95))

    print(f"[{name}] GNSS: {len(df_gnss)}行 / 加速度: {len(df_accelerometer)}行")

    data = list()

    step = int(100 * interval + 0.5)    # 加速度のサンプル数
    stride = int(10 * interval + 0.5)   # GNSSのサンプル数

    # --- GNSSデータの区間平均 ---
    for i in range(0, len(df_gnss) - stride, stride):
        slice10 = df_gnss.iloc[i:i + stride]
        s = slice10.loc[:, 'activity']

        Velocity_ave = slice10['Velocity'].mean()
        Acceleration_ave = slice10['Acceleration'].mean()
        Latitude_ave = slice10['Latitude'].mean()
        Longitude_ave = slice10['Longitude'].mean()

        rawdata = [normalize_label(s.iloc[0]), Velocity_ave, Acceleration_ave,
                   Latitude_ave, Longitude_ave]
        
        data.append(rawdata)

    min_len = min(len(data), len(range(0, len(df_accelerometer) - step, step)))
    data = data[:min_len]

    # --- 加速度データの区間平均 ---
    for i in range(0, min_len * step, step):
        slice100 = df_accelerometer.iloc[i:i + step]

        Orientation_forward_ave = slice100['imuOrientation.forward'].mean()
        Orientation_side_ave = slice100['imuOrientation.side'].mean()
        Facing_ave = slice100['Facing'].mean()
        Forward_ave = slice100['Acceleration.forward'].mean()
        Side_ave = slice100['Acceleration.side'].mean()
        Up_ave = slice100['Acceleration.up'].mean()
        imuFoward_ave = slice100['imuAcceleration.forward'].mean()
        imuSide_ave = slice100['imuAcceleration.side'].mean()
        imuUp_ave = slice100['imuAcceleration.up'].mean()
        Roll_ave = slice100['Rotation.roll'].mean()
        Pitch_ave = slice100['Rotation.pitch'].mean()
        Yaw_ave = slice100['Rotation.yaw'].mean()

        data[i // step].extend([Orientation_forward_ave, Orientation_side_ave,
                                Facing_ave, Forward_ave, Side_ave, Up_ave,
                                imuFoward_ave, imuSide_ave, imuUp_ave,
                                Roll_ave, Pitch_ave, Yaw_ave, name])

    # キャッシュ保存
    with open(pickle_file, "wb") as f:
        pickle.dump(data, f)

    if panda:
        return get_df(data, includeNan=includeNan, fillNan=fillNan)
    else:
        return get_raw(data)


def load_all(panda=True, includeNan=False, fillNan=False, interval=1, skip=[]):
    names = list(set(FILES.keys()) - set(skip))

    if panda:
        df_list = [load_data(name, panda=True, includeNan=includeNan,
                             fillNan=fillNan, interval=interval)
                   for name in names]
        return pd.concat(df_list)
    else:
        data = []
        label = []
        for name in names:
            d, l = load_data(name, panda=False, interval=interval)
            data += d
            label += l
        return data, label
    