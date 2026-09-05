"""
プレーの補完

"""

import json
from os import sep
from tkinter import Y
from sklearn.decomposition import PCA #主成分分析器
import scipy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from copy import deepcopy
from dataloader import load_data    #データローダをimport
from sklearn.metrics import accuracy_score  #ラベルの実際値と予測値を正解率として出力してくれる関数
from sklearn.metrics import precision_score #適合率
from sklearn.metrics import recall_score #再現率
from sklearn.metrics import f1_score # F値
from itertools import product
import numpy as np
from sklearn.model_selection import GridSearchCV
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.datasets import load_basic_motions
from random import sample
import matplotlib.pyplot as plt
from utils import positiveAnalysis #真値と予測値をプロット
from utils import sub_process,gridsearch,CompletionAndGrouping,completion,grouping,evaluation
import yaml
import sys
import os
import datetime


# パラメータの読み込み
if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
  setting_file = sys.argv[1]
else:
  print("パラメータファイルが存在しません！")
  exit()

with open(setting_file, 'r') as f:
  settings = yaml.load(f)


def main_process(config):
  name        = config["names"]
  columns     = config["columns"]
  classifier  = config["models"]
  classifier2 = config["model2"]
  usePCA      = config["PCA"]
  PCNum       = config["PCNum"]
  seeds       = config["seeds"]
  two_level_classification = config["two_level_classification"]
  use_completion  = config["completion"]
  use_grouping    = config["grouping"]
  dataset = config["datasets"]

  export_data = {}

  # 初期値を変化させる
  for seed in range(seeds):
    print("シード:", seed)
    models1 = {}
    models2 = {}
    
    # 各試合を除いて学習する
    for name1 in dataset:
      X_train, y_train = sub_process(load_data(name, panda=True, includeNan=True, fillNan=True, interval=1.0, skip=[name1])[columns],usePCA,PCNum)

      print(f"学習中: {name1} out")

      # 二段階分類で学習
      if two_level_classification:
        # モデル1: 非プレーとプレーの2ラベル
        X_train_model1 = deepcopy(X_train)
        y_train_model1 = deepcopy(y_train)

        y_train_model1[(y_train_model1 == "scrum") + (y_train_model1 == "Not_Scrum")] = "Not_Nan"

        # モデル2: スクラムとその他のプレーの2ラベル(非プレーを除外)
        X_train_model2 = X_train[(y_train == "scrum") + (y_train == "Not_Scrum")]
        y_train_model2 = y_train[(y_train == "scrum") + (y_train == "Not_Scrum")]


        if classifier == "SVC":
          model1 = SVC(probability=True,random_state=seed)    #分類器の作成
        else:
          model1 = RandomForestClassifier(random_state=seed)    #分類器の作成

        if classifier2 == "SVC":
          model2 = SVC(probability=True,random_state=seed)    #分類器の作成
        else:
          model2 = RandomForestClassifier(random_state=seed)    #分類器の作成

        model1.fit(X_train_model1, y_train_model1) #学習
        model2.fit(X_train_model2, y_train_model2) #Nanを抜いたデータで学習/ Scrumと非Scrum

        models1[name1] = model1
        models2[name1] = model2

      else:
        if classifier == SVC:
          model1 = SVC(probability=True,random_state=seed)    #分類器の作成
        else:
          model1 = RandomForestClassifier(random_state=seed)    #分類器の作成
        model1.fit(X_train, y_train) #学習
        models1[name1]  = model1



    # 評価をする
    if two_level_classification:
      thresh_range = range(11)
    else:
      thresh_range = (-1, )
    
    for thre1 in thresh_range:
      thresh1 = thre1 / 10
      for thre2 in thresh_range:
        thresh2 = thre2 / 10

        y_test_all = []
        y_pred_all = []

        print("閾値1:", thresh1)
        print("閾値2:", thresh2)

        # 各試合データを用いて評価する
        for name1 in dataset:
          X_test, y_test = sub_process(load_data(name1, panda=True, includeNan=True, fillNan=True, interval=1.0)[columns],usePCA,PCNum)

          if two_level_classification:
            model1 = models1[name1]
            model2 = models2[name1]

            y_pred = []
            # print(model1.classes_, model2.classes_)

            # 予測
            y_pred_model1 = [model1.classes_[x] for x in (model1.predict_proba(X_test)[:,1] >= thresh1).astype(int)]
            y_pred_model2 = [model2.classes_[x] for x in (model2.predict_proba(X_test)[:,1] >= thresh2).astype(int)]

            # 2つのモデルの予測結果をまとめて3ラベル化
            for i in range(len(y_pred_model1)):
              if y_pred_model1[i] == "Nan" : 
                y_pred.append(y_pred_model1[i])
              else :
                y_pred.append(y_pred_model2[i])

            # 全試合予測結果に加える
            y_pred_all += y_pred
            y_test_all += y_test.tolist()

          else:
            model1 = models1[name1]
            y_pred = model1.predict(X_test)
            y_pred_all += y_pred.tolist()
            y_test_all += y_test.tolist()


        # 全試合予測結果を用いて評価結果を求める
        if use_completion and use_grouping:
          # 補完・結合あり
          result = CompletionAndGrouping(y_pred_all, y_test_all)
        elif use_completion:
          # 補完のみ
          y_test_all[(y_test_all != 'scrum')] = 'Nan'
          y_pred_all[(y_pred_all != 'scrum')] = 'Nan'
          y_test_all = completion(y_test_all)
          y_pred_all = completion(y_pred_all)
          
          # 結合なし
          y_true_scrum = [int(x == 'scrum') for x in y_test_all]
          y_pred_scrum = [int(x == 'scrum') for x in y_pred_all]

          result = {
            "scrum": [
              accuracy_score(y_true_scrum, y_pred_scrum), 
              precision_score(y_true_scrum, y_pred_scrum), 
                recall_score(y_true_scrum, y_pred_scrum), 
                    f1_score(y_true_scrum, y_pred_scrum)
              ]
          }
        elif use_grouping:
          # 結合のみ
          result = evaluation(grouping(y_pred_all), grouping(y_test_all))
        else:
          # 補完・結合なし
          y_test_all[(y_test_all != 'scrum')] = 'Nan'
          y_pred_all[(y_pred_all != 'scrum')] = 'Nan'
          
          # 結合なし
          y_true_scrum = [int(x == 'scrum') for x in y_test_all]
          y_pred_scrum = [int(x == 'scrum') for x in y_pred_all]

          result = {
            "scrum": [
              accuracy_score(y_true_scrum, y_pred_scrum), 
              precision_score(y_true_scrum, y_pred_scrum), 
                recall_score(y_true_scrum, y_pred_scrum), 
                    f1_score(y_true_scrum, y_pred_scrum)
              ]
          }

        # ここからは同じ
        if not (thresh1, thresh2) in export_data:
          export_data[(thresh1, thresh2)] = {'scrum':[]}

        export_data[(thresh1, thresh2)]['scrum'].append(result['scrum'])

  
  # 全初期値・全閾値・全試合の結果を保存用の形式に変換する
  final_data = []
  for key, val in export_data.items():
    data = {}
    # for label in ['scrum', 'Not_Scrum']:
    for label in ['scrum']:
    # for label in ['Not_Scrum']:
      tmp = [[] for _ in range(4)]
      # 転置
      for result in val[label]:
        for i, v in enumerate(result):
          tmp[i].append(v)
      data[label] = tmp
    final_data.append({'thresh': key, 'result': data})

  # 結果を保存する
  with open("./結果/result_" + datetime.datetime.now().strftime("%y%m%d%H%M%S") + '.json',"w") as f:
    json.dump({"config": config, "result": final_data}, f)
  return 0


if __name__ == "__main__":
  gridsearch(main_process, settings)