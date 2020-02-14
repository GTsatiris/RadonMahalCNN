import numpy as np
import random

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from sklearn import preprocessing

import os
import progressbar

DATASIZE = 17847

ALL_DATA = np.zeros((DATASIZE, 180, 180, 1))
ALL_CLS = np.zeros(DATASIZE, dtype=int)

directory = os.listdir('Data/NPZ')
index = 0

print('Loading data...')
bar = progressbar.ProgressBar(maxval=DATASIZE, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for sdir in directory:
    sdirSTR = 'Data/NPZ/' + sdir
    subdir = os.listdir(sdirSTR)
    for fname in subdir:
        # print(sdirSTR + '/' +fname)
        # print(index)
        data = np.load(sdirSTR + '/' +fname)
        # ALL_DATA[index, :, :, 0] = preprocessing.normalize(data['sino'])
        ALL_DATA[index, :, :, 0] = data['sino']
        ALL_CLS[index] = data['clNum'] - 1
        bar.update(index + 1)
        index = index + 1
bar.finish()

TESTSIZE = int(DATASIZE * 0.2)
testIndexes = random.sample(range(0, DATASIZE), TESTSIZE)

TRAINSIZE = DATASIZE - TESTSIZE

x_train = np.zeros((TRAINSIZE, 180, 180, 1))
y_train = np.zeros(TRAINSIZE, dtype=int)
x_test = np.zeros((TESTSIZE, 180, 180, 1))
y_test = np.zeros(TESTSIZE, dtype=int)

trainIdx = 0
testIdx = 0

print('Partitioning data...')
bar = progressbar.ProgressBar(maxval=DATASIZE, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for idx in range(0, DATASIZE):
    if idx in testIndexes:
        x_test[testIdx, :, :, 0] = ALL_DATA[idx, :, :, 0]
        y_test[testIdx] = ALL_CLS[idx]
        testIdx = testIdx + 1
    else:
        x_train[trainIdx, :, :, 0] = ALL_DATA[idx, :, :, 0]
        y_train[trainIdx] = ALL_CLS[idx]
        trainIdx = trainIdx + 1
    bar.update(idx + 1)
bar.finish()
y_train_C = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_C = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

# sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adadelta(), metrics=['sparse_categorical_accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])

# history = model.fit(x_train, y_train, batch_size=64, epochs=10)
# test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=64)

history = model.fit(x_train, y_train_C, batch_size=32, epochs=10)
print('Testing...')
test_loss, test_acc = model.evaluate(x_test, y_test_C, batch_size=32)
