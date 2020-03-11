#!/usr/bin/env python
# coding: utf-8

# # Libs

# In[ ]:


import os
import re
import sys
import glob
import time
import datetime
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("d:\softwares\python36\lib\site-packages")
from easyeda import eda
from geohash import encode
from geopy.distance import great_circle

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, LeakyReLU, Input
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLD,categorical_crossentropy
from tensorflow.keras.utils import normalize

import lightgbm as lgb


# # Load Data

# In[ ]:


filedir = glob.glob(pathname='../Data/*.csv')
filedir


# In[ ]:


submit_table = pd.read_csv('../Data/submit_example/test_submit_example.csv', header=None)
submit_data = submit_table.copy()


# # Feature Engineering

# ## area info

# In[ ]:


# load data
area_passenger_info = pd.read_csv(filedir[1], header=None)
area_passenger_info.columns = ['areaIdx', 'areaName', 'areaType', 'centerLon', 'centerLat',
                               'gridLon', 'gridLat', 'coverage']
area_passenger_info.info()

# area type
# 交通设施：0-2，旅游景点：3，教育培训：4，购物：5，医疗：6，运动健身：7
areaTypes = area_passenger_info['areaType'].unique()
normalTypes = {'旅游景点':3,'教育培训':4,'购物':5,'医疗':6,'运动健身':7}
type_to_idx = {}
idx = 0
for item in areaTypes:
    preType = re.match("(.*);(.*)",item)[1]
    if  preType == '交通设施':
        type_to_idx[item] = idx
        idx += 1
    elif preType in normalTypes.keys():
        type_to_idx[item] = normalTypes[preType]
    else:
        print("this type does not exist.")

area_passenger_info['areaType'] = area_passenger_info['areaType'].replace(type_to_idx)
area_passenger_info['radius'] = np.sqrt(area_passenger_info['coverage'])
area_passenger_info['coverage'] = area_passenger_info['coverage'] / 4e+4


# In[ ]:


area_passenger_info['coord'] = area_passenger_info.apply(lambda x: (x['centerLat'],
                                                                    x['centerLon']),
                                                         axis=1)
CBDcoord = (39.91178273927437, 116.4015680859375)
area_passenger_info['cbdDist'] = area_passenger_info['coord'].map(
    lambda x: great_circle(CBDcoord, x).km)


# In[ ]:


num_areas = len(area_passenger_info)
distance = np.zeros((num_areas, num_areas))
for i in range(num_areas):
    a = area_passenger_info['coord'][int(i)]
    for j in range(num_areas):
        if j >= i:
            break
        b = area_passenger_info['coord'][int(j)]
        distance[i, j] = great_circle(a,b).km
area_distance = distance.T + distance


# In[ ]:


area_passenger_info['avgDistance'] = area_distance.sum(axis = 1) / (num_areas - 1)

std = np.zeros((num_areas,1))
num_inc_areas = np.zeros((num_areas,1))
closest_five = np.zeros((num_areas,5))
closest_five_dist = np.zeros((num_areas,5))
for i in range(num_areas):
    # rm zeros
    base_rm = list(set(area_distance[i,:]) - {0})
    radius = area_passenger_info['radius'].iloc[i]
    num_inc_areas[i,:] = np.sum(base_rm <= radius / 1000)
    std = np.std(base_rm)
    for j in range(5):
        closest_five[i,j] = np.argmin(base_rm)
        closest_five_dist[i,j] = np.min(base_rm)
        base_rm.remove(np.min(base_rm))

area_passenger_info['stdDistance'] = std
area_passenger_info['numIncludeAreas'] = num_inc_areas
for j in range(5):
    area_passenger_info['closestNo_' + str(j + 1)] = closest_five[:,j]
    area_passenger_info['closestDistNo_' + str(j + 1)] = closest_five_dist[:,j]


# In[ ]:


area_passenger_info.head()


# ## index-stats embedding

# In[ ]:


area_passenger_ind = pd.read_csv(filedir[0],header = None)
area_passenger_ind.columns = ['areaIdx','datetime','Density']
area_passenger_ind['datetime'] = pd.to_datetime(area_passenger_ind['datetime'],format="%Y%m%d%H")
area_passenger_ind.info()


# In[ ]:


area_passenger_ind['ToD'] = area_passenger_ind['datetime'].map(lambda x: x.hour)
area_passenger_ind['DoW'] = area_passenger_ind['datetime'].map(lambda x: x.weekday())
embed_data = pd.merge(area_passenger_ind, area_passenger_info, on='areaIdx')
area_passenger_ind = pd.merge(area_passenger_ind,
                              area_passenger_info[['areaIdx', 'areaType']],
                              on='areaIdx')


def trend(x):
    return np.mean(pd.Series(x).diff().fillna(0))


area_type_label = area_passenger_ind.pivot_table(index='areaType',
                                                 columns='ToD',
                                                 values='Density',
                                                 aggfunc=['mean', 'std', 'median', trend])
area_passenger_info = pd.merge(area_passenger_info,
                               area_type_label,
                               on='areaType')
area_passenger_ind.drop('areaType', axis=1, inplace=True)
embed_label = area_passenger_ind.pivot_table(index='areaIdx',
                                             columns='ToD',
                                             values='Density',
                                             aggfunc=['mean', 'std', 'median', np.ptp, trend])


# In[ ]:


def get_embedding(embedding_dim, batch_size, epochs):
    # build model
    x = Input(shape=(1,))
    o = Embedding(input_dim=997, output_dim=embedding_dim,
                  embeddings_initializer=he_normal(), name='embedding')(x)
    h = Dense(128, use_bias=False,
              kernel_initializer=he_normal(), activation='relu')(o)
    h = Dense(24 * 5, use_bias=False,
              kernel_initializer=he_normal(), activation='relu')(o)
    model = Model(inputs=x, outputs=h)
    model.compile(loss='mse', optimizer=Adam(3e-4))
    
    # train embedding weights
    hist = model.fit(np.arange(0, 997).reshape(-1, 1), normalize(embed_label.values),
                 batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
    
    # output embedding vector
    areaEmbedding = model.get_weights()[0]
    
    return areaEmbedding, hist


# In[ ]:


embedding_dim = 20
areaEmbedding, trainingLog = get_embedding(embedding_dim=embedding_dim,
                                           batch_size=16,epochs=500)


# ## Historic index (same area)

# In[ ]:


def get_hitoric_index(area_passenger_ind, window_size, num_samples, num_areas, num_days):
    
    # init
    historicIndex = np.zeros((num_samples, window_size))
    sample_idx = 0
    sp = time.time()
    
    # main loop
    for area_idx in range(1, num_areas + 1):
        if area_idx % 200 == 0:
            print("[Area-{:d}] started, duration: {:.1f} sec.".format(area_idx, time.time() - sp))
        area_df = area_passenger_ind[area_passenger_ind.areaIdx == area_idx]
        for i in range(24 * num_days - window_size):
            historicIndex[sample_idx] = area_df['Density'].values[i:i + window_size]
            sample_idx += 1
    
    return historicIndex


# In[ ]:


# params
window_size = 8
num_areas = 997
num_days = 30
num_samples = (24 * num_days - window_size)  * num_areas

# get historic index
historicIndex = get_hitoric_index(area_passenger_ind,
                                  window_size=window_size,
                                  num_samples=num_samples,
                                  num_areas=num_areas,
                                  num_days=num_days)


# In[ ]:


histMean = historicIndex.mean(axis=1)
histStd = historicIndex.std(axis=1)
histMedian = np.median(historicIndex, axis=1)
histPtp = np.ptp(historicIndex, axis=1)
histDiff = historicIndex[:,1:] - historicIndex[:,:(window_size - 1)]


# In[ ]:


historicIndexDf = pd.DataFrame()

for col in range(window_size):
    historicIndexDf['historic_' + str(col)] = historicIndex[:,col]

for col in range(window_size - 1):
    historicIndexDf['historic_diff_' + str(col)] = histDiff[:,col]

historicIndexDf['histMean'] = histMean
historicIndexDf['histStd'] = histStd
historicIndexDf['histMedian'] = histMedian
historicIndexDf['histPtp'] = histPtp


# ## Concat

# In[ ]:


areaEmbeddingDf = pd.DataFrame(np.arange(1,num_areas + 1),columns=['areaIdx'])
for col in range(embedding_dim):
    areaEmbeddingDf["embedding_" + str(col)] = areaEmbedding[:,col]


# In[ ]:


areaAttr = pd.merge(area_passenger_info, areaEmbeddingDf, on='areaIdx')
areaAttr.drop(['areaName','coord'], axis=1, inplace=True)


# In[ ]:


dfs = [area_passenger_ind, areaAttr, embed_label.reset_index()]
AreaDensity = reduce(lambda a,b:pd.merge(a,b,on='areaIdx'),dfs)
AreaDensity.drop("datetime",axis=1,inplace=True)


# # Build Dataset

# In[ ]:


# init
X_attr = np.zeros((num_samples,AreaDensity.shape[1]))
sample_idx = 0
sp = time.time()

# main loop
for area_idx in range(1, num_areas + 1):
    if area_idx % 200 == 0:
        print("[Area-{:d}] started, duration: {:.1f} sec.".format(area_idx, time.time() - sp))
    area_df = AreaDensity[AreaDensity.areaIdx == area_idx]
    for i in range(window_size, 24 * num_days):
        X_attr[sample_idx] = area_df.values[i,:]
        sample_idx += 1


# In[ ]:


X = pd.concat((pd.DataFrame(X_attr, columns=AreaDensity.columns),
               historicIndexDf), axis=1)
del X_attr, AreaDensity
Y_data = X['Density']
X_data = X.drop(['Density'], axis=1)
del X


# # LightGBM

# In[ ]:


params = {
    "objective":"regression",
    "num_rounds":10000,
    "learning_rate":0.01,
    "max_depth":9,
    "num_leaves":100,
    "feature_fraction":0.8,
    "verbose":2
}


# In[ ]:


def score(y_pred, y_test):
    rmse = np.sqrt(np.mean(np.square(y_pred - y_test)))
    return 1/(1 + rmse)


# In[ ]:


kf = KFold(n_splits=5,shuffle=True)
idx = 0
for train_idx, test_idx in kf.split(X_data):
    # dataset split
    X_train, y_train = X_data.iloc[train_idx], Y_data.iloc[train_idx]
    X_test, y_test = X_data.iloc[test_idx], Y_data.iloc[test_idx]
    
    # train
    categorical_features = ['areaIdx','ToD', 'DoW', 'areaType',
                            'closestNo_1','closestNo_2','closestNo_3',
                            'closestNo_4','closestNo_5']
    train_data = lgb.Dataset(X_train,y_train,categorical_feature=categorical_features)
    test_data = lgb.Dataset(X_test,y_test,reference=train_data)
    
    gbm = lgb.train(params,train_data)
    y_pred = gbm.predict(X_test,num_iteration=gbm.best_iteration)
    print("[CV-{:d}] score: {:.4f}".format(idx, score(y_pred,y_test)))
    idx += 1


# In[ ]:


lgb.plot_importance(gbm,max_num_features = 30, height = 0.5,figsize=(10,8),grid = False)


# # Test Auto-aggressive prediction

# In[ ]:


X_test = submit_table.copy()
X_test.columns = ['areaIdx','datetime','predIndex']

X_test['datetime'] = pd.to_datetime(X_test['datetime'],format="%Y%m%d%H")
X_test['ToD'] = X_test['datetime'].map(lambda x: x.hour)
X_test['DoW'] = X_test['datetime'].map(lambda x: x.weekday())

dfs = [X_test,area_passenger_info,areaEmbeddingDf]
X_test = reduce(lambda a,b:pd.merge(a,b,on='areaIdx'),dfs)


# In[ ]:


def get_historic_feature(values, window_size):
    historic_ = values[-window_size:]
    historic_diff_ = np.array(historic_[-window_size + 1:]) - np.array(historic_[-window_size:-1])
    histMean = np.mean(historic_)
    histStd = np.std(historic_)
    histMedian = np.median(historic_)
    histPtp = np.ptp(historic_)
    return historic_ + historic_diff_.tolist() + [histMean,histStd,histMedian,histPtp]


# In[ ]:


test_input = X_test.drop(["datetime","predIndex","areaName"],axis = 1)
sample_idx = 0
sp = time.time()
for areaIdx in range(1, 1 + num_areas):
    X_test_area = X_test[X_test.areaIdx == areaIdx]
    histValues = area_passenger_ind[area_passenger_ind.areaIdx == areaIdx]['Density'].values.tolist()
    for i in range(len(X_test_area)):
        # predict
        histFeat = get_historic_feature(histValues,window_size=window_size)
        test_sample_input = test_input.iloc[i,:].tolist() + histFeat + test_svd
        pred_value = gbm.predict([test_sample_input])[0]
        
        # update submit file
        submit_table.iloc[sample_idx,2] = pred_value
        
        # update aggressive base
        histValues.append(pred_value)
        sample_idx += 1
        
    print("[Area-{:d}] Finished. Duration: {:.1f} sec.".format(areaIdx,time.time() - sp))


# In[ ]:


submit_table.to_csv('../Data/submit_files/',header=None,index=None)

