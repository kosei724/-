"""
データを読み込むデータローダーのファイル
"""
# データを読み込むデータローダーを作るファイル
from statistics import variance
import numpy as np
import pandas as pd
import os   #フォルダの中に入っているファイルを読み込める
import pickle
import sys
import pathlib

dataset_path = pathlib.Path('./試合データセット')
gnss_dir = 'GNSSファイル'
# gnss_dir = 'GNSSファイル(相対位置)'
acceleration_dir = '加速度ファイル'

def get_df(data,includeNan=False,fillNan=False):
    columns = ["label", "Velocity", "Acceleration", "Latitude", "Longitude", "imuOrientation.forward", "imuOrientation.side","Facing","Acceleration.forward", "Acceleration.side","Acceleration.up", "imuAcceleration.forward","imuAcceleration.side","imuAcceleration.up", "Rotation.roll", "Rotation.pitch", "Rotation.yaw", "Variance_all", "Variance_Atk"]

    df = pd.DataFrame(data, columns=columns)
    if not includeNan:
      df2 = df.dropna(how='any').dropna(how='any', axis=1)
    else:
      df2 = df
    # print(df2)

    if fillNan: #Nanを埋める
      df3 = df2.fillna({"label":"Nan"})
    else:
      df3 = df2

    return df3

def load_datetime(name):
  # 加速度データ読み込む
  if name == "toshiba":
    acceleration_file = 'accelerometer_タグ付け_Toshiba.csv'
  if name == "toyota":
    acceleration_file = 'accelerometer_タグ付け_Toyota.csv'
  if name == "ricoh":
    acceleration_file = 'accelerometer_タグ付け_Ricoh.csv' 
  if name == "toshiba_6DK":
    acceleration_file = 'accelerometer_タグ付け_Toshiba_6DK.csv'
  if name == "toshiba_7OSUGI":
    acceleration_file = 'accelerometer_タグ付け_Toshiba_7OSUGI.csv'
  if name == "scrum_practice":
    acceleration_file = 'accelerometer_タグ付け_Scrum_Practice.csv'

  df_accelerometer = pd.read_csv(dataset_path / acceleration_dir / acceleration_file)
  df_accelerometer = df_accelerometer.drop(range(95))   #1区間減らす(100行)


  return df_accelerometer["Date/Time"].iloc[0]


def get_raw(data):
    x = []        #データ(入力:x)
    label = []    #ラベル(出力:y)

    for i in range(len(data)):    #データをlabel(y:出力)とX(入力)に分ける
      x.append(data[i][1:])       #xに1から最後までのデータを追加
      label.append(data[i][0])    #labelに0番目のデータを追加

    return x, label

# データを読み込む関数
def load_data(name, panda=True, includeNan=False, fillNan=False, interval=1, skip=[]):
  if name == "all":
    return load_all(panda, includeNan, fillNan, interval,skip)
  
  pickle_file = f"./前処理済みデータ/{name}_interval_{interval}.pickle"
  # print(sys.argv[1])

  if os.path.exists(pickle_file):
    # print("pickle file found. loading...")
    with open(pickle_file, "rb") as f:
      data = pickle.load(f)

    if panda:
      return get_df(data, includeNan=includeNan, fillNan=fillNan)
    else:
      return get_raw(data)


  # GNSSデータ読み込む
  if name == "toshiba":
    gnss_file = 'gnss_タグ付け_Toshiba.csv'
  if name == "toyota":
    gnss_file = 'gnss_タグ付け_Toyota.csv'
  if name == "ricoh":
    gnss_file = 'gnss_タグ付け_Ricoh.csv' 
  if name == "toshiba_6DK":
    gnss_file = 'gnss_タグ付け_Toshiba_6DK.csv'
  if name == "toshiba_7OSUGI":
    gnss_file = 'gnss_タグ付け_Toshiba_7OSUGI.csv'
  if name == "scrum_practice":
    gnss_file = 'gnss_タグ付け_Scrum_Practice.csv'

  df_gnss = pd.read_csv(dataset_path / gnss_dir / gnss_file)
  df_gnss = df_gnss.drop(range(10))     #1区間減らす(10行)

  # 加速度データ読み込む
  if name == "toshiba":
    acceleration_file = 'accelerometer_タグ付け_Toshiba.csv'
  if name == "toyota":
    acceleration_file = 'accelerometer_タグ付け_Toyota.csv'
  if name == "ricoh":
    acceleration_file = 'accelerometer_タグ付け_Ricoh.csv' 
  if name == "toshiba_6DK":
    acceleration_file = 'accelerometer_タグ付け_Toshiba_6DK.csv'
  if name == "toshiba_7OSUGI":
    acceleration_file = 'accelerometer_タグ付け_Toshiba_7OSUGI.csv'
  if name == "scrum_practice":
    acceleration_file = 'accelerometer_タグ付け_Scrum_Practice.csv'

  df_accelerometer = pd.read_csv(dataset_path / acceleration_dir / acceleration_file)
  df_accelerometer = df_accelerometer.drop(range(95))   #1区間減らす(100行)

  print(df_gnss)
  print(df_accelerometer)


  #リストの中にリストを追加する
  data = list() #空のリスト
  
  step = int(100 * interval + 0.5) #加速度(サンプル数)
  stride = int(10 * interval + 0.5) #GNSS(サンプル数)

  # GNSSデータ
  for i in range(0,len(df_gnss)-stride,stride):       #0~最後の行-10, 10刻み
    slice10 = df_gnss.iloc[i:i+stride]    #10区切り
    t = slice10.loc[:,['Velocity','Acceleration','Latitude','Longitude']]   #各列を抽出
    s = slice10.loc[:,'activity']   #列を抽出
    # print(s) 
    # print(s.iloc[0])    #ラベル

    #平均(10区切りデータ)
    Velocity_ave = slice10['Velocity'].mean()
    Acceleration_ave = slice10['Acceleration'].mean()
    Latitude_ave = slice10['Latitude'].mean()
    Longitude_ave = slice10['Longitude'].mean()

    # print(slice10['Velocity'])
    # # print(Velocity_ave)
    # # if i == 10:
    # #   raise ValueError()

    #中身のあるリスト(平均データ)
    rawdata = [s.iloc[0], Velocity_ave, Acceleration_ave, Latitude_ave, Longitude_ave]

    #データに追加する(1行ずつ)
    data.append(rawdata)    #要素を追加する
  
  min_len = min(len(data), len(range(0,len(df_accelerometer)-step,step)))
  data = data[:min_len]
  # print(len(data))
  # print(len(range(0,len(df_accelerometer)-500,100)))
  # raise ValueError()

  # 加速度データ
  for i in range(0, min_len*step, step):       #0~最後の行-100, 100刻み
    slice100 = df_accelerometer.iloc[i:i+step]    #100区切り
    # d = slice100.loc[:,['Acceleration.forward','imuOrientation.forward','imuOrientation.side','Facing','Acceleration.side','Acceleration.up','Rotation.roll','Rotation.pitch','Rotation.yaw']]    #各列を抽出
    h = slice100.loc[:,'activity']
    # print(h)

    print(h.iloc[0])

    #平均(100区切りデータ)
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

    #GNSSリストに加速度リストを結合(平均データ)
    data[i//step].extend([Orientation_forward_ave,Orientation_side_ave,Facing_ave,Forward_ave,Side_ave,Up_ave,imuFoward_ave,imuSide_ave,imuUp_ave,Roll_ave,Pitch_ave,Yaw_ave])

    # raise ValueError()

  #みんなのGNSSデータ/7人のデータ
  #スクラムなどを確かめるため(分散)
  def variance(datadir, num):
    data2 = [[] for i in range(num)] #14の空リスト,1人データが少なかったため取り除いた
    j = 0

    for file in os.listdir(datadir):
      df_gnss = pd.read_csv(datadir+file, header=8)   #9行目から読み取る
      df_gnss = df_gnss.drop(range(10))
      print(df_gnss)
      for i in range(0,len(df_gnss)-stride,stride):       #0~最後の行-10, 10刻み
          slice10 = df_gnss.iloc[i:i+stride]    #10区切り
          z = slice10.loc[:,['Latitude','Longitude']]   #各列を抽出
          
          # print(s) 
          # print(s.iloc[0])    #ラベル

          #平均(10区切りデータ)
          Latitude_ave = slice10['Latitude'].mean()
          Longitude_ave = slice10['Longitude'].mean()

          # print(slice10['Velocity'])
          # # print(Velocity_ave)
          # # if i == 10:
          # #   raise ValueError()

          #中身のあるリスト(平均データ)
          rawdata2 = [Latitude_ave, Longitude_ave]

          #データに追加する(1行ずつ)
          data2[j].append(rawdata2)    #要素を追加する

      j += 1

    # data = data[:len(data2[0])]   #データを8069を前半の3009までに切り落としている
    # #data = data[:len(data)]   #データを8069(1試合分(後半含む))
    # print(len(data))      #データの長さ
    # print(len(data2[0]))  #みんなのデータの長さ
    # #raise ValueError()
    # print(data2[0][0])    #15個ある時の0人目のデータ0
    # print(np.array(data2).shape)

    print("hello")
    for i in range(num): #14人/7人
      data2[i] = data2[i][:len(data2[0])]
      print(len(data2[i]))

    variance_list = []
    
    #分散
    for i in range(len(data2[0])):
      variance_latitude = []
      variance_longitude = []

      for j in range(num):
        variance_latitude.append(data2[j][i][0])
        variance_longitude.append(data2[j][i][1]) 

      variance = np.var(variance_latitude) + np.var(variance_longitude)   #分散(固まっている、広がっているなど)＝値が1つ
      variance_list.append(variance)

    return variance_list

  #変更した（後で復活させて）
  if name == "toyota":
    Variance_all = variance("GPS/", 14)     #14人のToyotaデータ
    Variance_Attacker = variance("GPS_Attacker/", 7)    #7人のToyotaデータ
  elif name == "toshiba":
    Variance_all = variance("GPS_Toshiba/", 15)     #15人のToshibaデータ
    Variance_Attacker = variance("GPS_Toshiba_Attacker/", 8)    #8人のToshibaデータ
  else:
    Variance_all = [0.0, 1.0] * (len(data)//2)
    Variance_Attacker = [0.0, 1.0] * (len(data)//2)
    # Variance_all[0] = 1000000000.0
    # Variance_Attacker[0] = 10000000000.0


  data = data[:len(Variance_all)]       #Toyota14人
  for i in range(len(data)):
    data[i].append(Variance_all[i]) #ここでアタッカー結合している
    data[i].append(Variance_Attacker[i]) #ここでアタッカー結合している


  # label = []    #ラベル(出力:y)
  # x = []        #データ(入力:x)

  with open(pickle_file, "wb") as f:
    pickle.dump(data, f)

  if panda:
    return get_df(data, includeNan=includeNan, fillNan=fillNan)
  else:
    return get_raw(data)
  
def load_all(panda=True, includeNan=False, fillNan=False, interval=1, skip=[]):
  names = set(["toyota", "ricoh","toshiba_6DK","toshiba_7OSUGI", "toshiba","scrum_practice"]) - set(skip)
  names = list(names)
  # names = ["toyota", "ricoh","toshiba_6DK","toshiba_7OSUGI", "toshiba"]
  if panda:
    df_list = [load_data(name, panda=True, includeNan=includeNan, fillNan=fillNan, interval=interval) for name in names]
    return pd.concat(df_list)
  
  
  else:
    data = []
    label = []
    for name in names:
      d, l = load_data(name, panda=False, interval=interval)
      data += d
      label += l
    return data, label