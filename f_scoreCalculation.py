import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# DON'T NEED THEM NOW
import os
os.environ["PATH"] += os.pathsep + 'C:/GraphViz/bin/'
from tensorflow.keras.utils import plot_model
# from sklearn import preprocessing

import progressbar

DATASIZE = 17847
directory = os.listdir('Data/NPZ')

subj = ['s1_', 's2_', 's3_', 's4_', 's5_', 's6_']
frame20 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.']
frame30 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.', 'frame_21.', 'frame_22.', 'frame_23.', 'frame_24.', 'frame_25.', 'frame_26.', 'frame_27.', 'frame_28.', 'frame_29.', 'frame_30.']

# for subject in subj:
subject = 's1_'
bSize = 32

x_train_L = []
y_train = []

# data1 = np.load('Data/NPZ/' + directory[0] + '/frame_5.npz')
# data2 = np.load('Data/NPZ/' + directory[0] + '/frame_6.npz')
# data3 = np.load('Data/NPZ/' + directory[0] + '/frame_7.npz')

# data = np.dstack((data1['sino'], data2['sino'], data3['sino']))

# print(data.shape)

print('Loading and partitioning data...')
index = 0
bar = progressbar.ProgressBar(maxval=DATASIZE,
                                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for sdir in directory:
    sdirSTR = 'Data/NPZ/' + sdir
    subdir = os.listdir(sdirSTR)
    for fIdx in range(len(subdir)):
        # if sum(x in subdir[fIdx] for x in frame20) == 0:
        data1 = np.load(sdirSTR + '/' + subdir[fIdx])
        x_train_L.append(data1['sino'].flatten())
        # print(type(data1['sino'].flatten()))
        # print(data1['sino'].flatten().shape)
        y_train.append(data1['clNum'] - 1)
        bar.update(index + 1)
        index = index + 1
bar.finish()

x_train = np.asarray(x_train_L).reshape([len(x_train_L), 32400])
# y_train_C = tf.keras.utils.to_categorical(y_train, num_classes=10)

selector = SelectKBest(f_classif, k=1000)
selected_features = selector.fit_transform(x_train, y_train)

f_score_indexes = (-selector.scores_).argsort()[:1000]

print(type(f_score_indexes))