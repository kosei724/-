
"""
プレーの補完

"""


from sklearn.decomposition import PCA #主成分分析器
from itertools import product
import numpy as np
import matplotlib.pyplot as plt



def sub_process(Data,usePCA,PCNum):
  X = Data.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
  y = Data['label'].values   
  newX = X.values
  newy = y
  newy[(newy != "Nan") * (newy != "scrum")] = "Not_Scrum"

  #主成分分析の実行
  if usePCA:
    pca = PCA()
    pca.fit(newX)
    # データを主成分空間に写像
    newX = pca.transform(newX)
    newX = newX[:,:PCNum]

  return newX, newy





"""
gridsearch

"""
def gridsearch(main_process, settings):
  def mymap(key):
    return settings[key]
  print("******** parameters ********")
  for key, value in settings.items():
    print(key, ":", value)

#   keys = ["names", "models","PCA", "columns","PCNum"]
  keys = settings.keys()
  config_id = 1
  addScore = 0
  for values in product(*list(map(mymap, keys))):
    config = dict(zip(keys, values))
    print(f"********* config {config_id} *********")
    for key, value in config.items():
      if key == "columns":
        print(key, ":", "pattern"+str(settings["columns"].index(value) + 1))
      else:
        print(key, ":", value)
    addScore += main_process(config)
    config_id += 1
  addScore = addScore / (config_id - 1)


def CompletionAndGrouping(pred, test):
  pred = grouping(completion(pred))
  test = grouping(completion(test))

  result = evaluation(pred, test)
  return result
  

"""
補完

"""
def completion(value):
  interval = 15
  # interval =
  for i in range(len(value)-interval):
    if (value[i]) == "Nan":
      continue
    flag = False
    for j in range(i+interval, i, -1):
      # print(j)
      if(value[j] == value[i]):
        flag = True
      
      if(flag == True):
        value[j] = value[i]
  
  return value

"""
グループ化

"""
def grouping(value):
  result = []
  for i in range(1,len(value)):
    if (value[i] != value[i-1]) and (value[i] != "Nan"):
      result.append((value[i], i))
  return result

#比較(評価)
def evaluation(pred, test):
  pred = [tuple(list(x)+['p']) for x in pred]
  test = [tuple(list(x)+['t']) for x in test]
  # print(len(pred))

  value = pred + test
  value = sorted(value, key = lambda x: x[1])

  # modify the confusion matrix here
  n = len(value)
  allowed_diff = 15
  connected = set()
  confusion_matrix = {}

  data = {}
  data_match = {}
  data_match["scrum"] = []
  data_match["Not_Scrum"] = []
  
  for i in range(n):
    if not value[i][0] in data:
      data[value[i][0]] = []
    data[value[i][0]].append((value[i][1], value[i][2]))

    if value[i] in connected: continue

    j = i+1
    flg = False

    if not value[i][0] in confusion_matrix:
      confusion_matrix[value[i][0]] = {'TP':0, 'FP': 0, 'FN': 0}
      
    while j < n and value[j][1] - value[i][1] < allowed_diff:
      if (not value[j] in connected) and value[i][0] == value[j][0] and value[i][2] != value[j][2]:
        flg = True
        connected.add(value[i])
        connected.add(value[j])
        if not value[i][0] in data_match:
          data_match[value[i][0]] = []
        data_match[value[i][0]].append(((value[i][1], value[i][2]),(value[j][1], value[j][2])))
        confusion_matrix[value[i][0]]['TP'] += 1
        break
      j += 1
    
    # if pair not found
    if not flg:
      if value[i][2] == 'p':
        confusion_matrix[value[i][0]]['FP'] += 1
      else:
        confusion_matrix[value[i][0]]['FN'] += 1

  result = {}
  
  # for name in ['scrum', 'Not_Scrum']:
  for name in ['scrum']:
    TP = confusion_matrix[name]['TP'] + 0.0000001
    TN = 0 + 0.0000001
    FP = confusion_matrix[name]['FP'] + 0.0000001
    FN = confusion_matrix[name]['FN'] + 0.0000001

    accuracy_score = (TP+TN) / (TP+TN+FP+FN) 
    precision_score = TP / (TP+FP)
    recall_score = TP / (TP+FN)
    f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score)

    result[name] = [accuracy_score,precision_score,recall_score,f1_score]

    print("-"*15)
    print(name)
    print("-"*15)
    print('Test num: %d' % len([x for x in data[name] if x[1] == 't']))
    print('Pred num: %d' % len([x for x in data[name] if x[1] == 'p']))
    print('Accuracy: %.3f' % accuracy_score)  # y_test:ラベル/y_pred:予測ラベル
    print('Precision: %.3f' % precision_score)
    print('Recall: %.3f' % recall_score)
    print('F1: %.3f' % f1_score)
    print('Confusion Matrix:', confusion_matrix[name])

  name = "scrum"

  return result


"""
スタジアムの位置情報を相対位置に変換

"""

#スタジアムの緯度経度を座標変換の関数
# x,y は選手の緯度経度
# u,v 選手の相対位置
import numpy as np
import matplotlib.pyplot as plt
import math

def transformation(x,y,s,t,theta):
    u = math.cos(theta)*(x-s) + math.sin(theta)*(y-t)
    v = -math.sin(theta)*(x-s) + math.cos(theta)*(y-t)
    return u,v


num =  [[35.6731207,139.7178135],
        [35.6730763,139.7181535],
        [35.672575,139.7181525],
        [35.6726798,139.7183315],
        [35.6727412,139.7185735],
        [35.6724588,139.7180207],
        [35.672149,139.718265],
        [35.6722308,139.7185622],
        [35.6731572,139.7185252]]

def test():
    for i in range(9):
        x = num[i][1]
        y = num[i][0]
        #前半
        s1 = 139.718233
        t1 = 35.672040
        q1 = 139.718951
        r1 = 35.672273
        theta1 = math.atan2(r1-t1, q1-s1)
        # theta1 = 0.3759934020889578

        #後半
        s2 = 139.718460
        t2 = 35.673280
        q2 = 139.717742
        r2 = 35.673046
        theta2 = math.atan2(r2-t2, q2-s2)
        # theta2 = theta1 + 3.14159265359
        
        u1,v1 = transformation(x,y,s1,t1,theta1)
        u2,v2 = transformation(x,y,s2,t2,theta2)

        u1 = (u1 / 360) * 40000000 #メートルで表記
        u2 = (u2 / 360) * 40000000 #メートルで表記
        v1 = (v1 / 360) * 40000000 #メートルで表記
        v2 = (v2 / 360) * 40000000 #メートルで表記
        print(u1,v1,u2,v2)

        num[i][1] = u1
        num[i][0] = v1

        # print((theta1/3.14159265359)*180)
        # print(math.sin(theta1))


# test()

# num = np.array(num)

# plt.scatter(num.T[1], num.T[0])
# plt.show()


"""
真値と予測値をプロットするファイル

"""

import matplotlib.pyplot as plt



def positiveAnalysis(name, data, data_match):
  # 
  # (時刻, 真値('t')/予測値('p'))
  # data = [
  #   (8, 't'),
  #   (15,'p'),
  #   (21,'p'),
  #   (35,'t'),
  #   (44,'p'),
  #   (48,'t'),
  #   (69,'p'),
  #   (72,'t'),
  #   (75,'p'),
  #   (88,'p'),
  #   (92,'t'),
  #   (96,'p'),
  #   (99,'p'),
  # ]

  # data_match = [
  #   ((8, 't'),(15,'p')),
  #   ((35,'t'),(44,'p')),
  #   ((69,'p'),(72,'t')),
  #   ((88,'p'),(92,'t')),
  # ]

  # print(list(zip(*data)))
  a, b = list(zip(*data))
  b = list(map(lambda x: int(x == 'p'), b))
  # a = [3, 21, 43, 44, 30, 66, 72, 88, 92, 99]
  # b = [0,  1,  0,  1,  0,  1,  0,  1,  0,  0]

  # c = [((43, 0), (0, 1))]

  fig, ax = plt.subplots()

  ax.scatter(a, b,s=4)
  plt.rcParams['font.family'] = 'Hiragino Sans'
  # 太さ設定
  plt.rcParams['font.weight'] = 'bold'

  for line in data_match:
    c, d = list(zip(*line))
    d = list(map(lambda x: int(x == 'p'), d))
    ax.plot(c, d)

  ax.set_ylim(-1, 2)
  plt.yticks([])
  # ax.set_xlim(data[-1][0]+100, 0)
  ax.set_ylabel('<< Actual Positive                  Predicted Positive">>')
  ax.set_xlabel('Time (s)')
  ax.set_title(f'{name} Positive Analysis')

  plt.show()
