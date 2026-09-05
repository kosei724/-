from cmath import nan
import pandas as pd
import numpy as np
import math


#Toyota
# df_gnss = pd.read_csv('./試合データセット/GPSファイル/gnss_タグ付け_Toyota.csv')   
# df_accelerometer = pd.read_csv('./試合データセット/加速度ファイル/accelerometer_タグ付け_Toyota.csv')

#Toshiba
# df_gnss = pd.read_csv('./試合データセット/GPSファイル/gnss_タグ付け_Toshiba.csv')   
# df_accelerometer = pd.read_csv('./試合データセット/加速度ファイル/accelerometer_タグ付け_Toshiba.csv')

#Ricoh
# df_gnss = pd.read_csv('./試合データセット/GPSファイル/gnss_タグ付け_Ricoh.csv')
# df_accelerometer = pd.read_csv('./試合データセット/加速度ファイル/accelerometer_タグ付け_Ricoh.csv')

#Toshiba(6DK)
df_gnss = pd.read_csv('./試合データセット/GPSファイル/gnss_タグ付け_Toshiba_6DK.csv')   
df_accelerometer = pd.read_csv('./試合データセット/加速度ファイル/accelerometer_タグ付け_Toshiba_6DK.csv')

# #Toshiba(7OSUGI)
#df_gnss = pd.read_csv('./試合データセット/GPSファイル/gnss_タグ付け_Toshiba_7OSUGI.csv')
#df_accelerometer = pd.read_csv('./試合データセット/加速度ファイル/accelerometer_タグ付け_Toshiba_7OSUGI.csv')

print(df_gnss)
print(df_accelerometer)

def count_activity(df):     #関数定義 (引数:データフレームワーク)
  count_dict = {'scrum':0}

  prev_act = nan  

  for act in df['activity']:
    if not type(act) is str:      #actがstringではない場合(Nan)
      prev_act = act
      continue
    
    if type(act) is str and  not type(prev_act) is str:   #actがstringの場合 & 前のactがstringではない場合
      if not act in count_dict:     #count_dictにactが含まれていない場合に新規に追加する
        count_dict[act] = 1
      else:
        count_dict[act] += 1        #追加している
    
    prev_act = act      #prev_actを更新する

  # print(count_dict)

  return count_dict     #戻り値

for column in df_gnss:  #列を回している
  print(column)

print("GNSSデータ:")
r = count_activity(df_gnss)   #関数呼び出し(使う)
print(r)
print("加速度データ:")
s = count_activity(df_accelerometer)  #関数呼び出し(使う)
print(s)