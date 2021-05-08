import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

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
subject = 's2_'
bSize = 32

x_train_L = []
y_train = []
x_test_L = []
y_test = []

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
    if subject in sdirSTR:
        for fIdx in range(len(subdir) - 2):
            # if sum(x in subdir[fIdx] for x in frame20) != 0:
            data1 = np.load(sdirSTR + '/' + subdir[fIdx])
            data2 = np.load(sdirSTR + '/' + subdir[fIdx + 1])
            data3 = np.load(sdirSTR + '/' + subdir[fIdx + 2])
            x_test_L.append(np.dstack((data1['sino'], data2['sino'], data3['sino'])))
            y_test.append([data1['clNum'] - 1])
            bar.update(index + 1)
            index = index + 1
    else:
        for fIdx in range(len(subdir) - 2):
            # if sum(x in subdir[fIdx] for x in frame20) == 0:
            data1 = np.load(sdirSTR + '/' + subdir[fIdx])
            data2 = np.load(sdirSTR + '/' + subdir[fIdx + 1])
            data3 = np.load(sdirSTR + '/' + subdir[fIdx + 2])
            x_train_L.append(np.dstack((data1['sino'], data2['sino'], data3['sino'])))
            y_train.append(data1['clNum'] - 1)
            bar.update(index + 1)
            index = index + 1
bar.finish()
x_train = np.asarray(x_train_L).reshape([len(x_train_L), 180, 180, 3])
x_test = np.asarray(x_test_L).reshape([len(x_test_L), 180, 180, 3])

y_train_C = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_C = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = tf.keras.applications.MobileNetV2(
    input_shape=(180, 180, 3),
    alpha=1.0,
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=10,
    classifier_activation="softmax"
)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=['categorical_accuracy'])

history = model.fit(x_train, y_train_C, batch_size=bSize, epochs=45)
print('Testing...')
test_loss, test_acc = model.evaluate(x_test, y_test_C, batch_size=bSize)

model.save('MOB_with_' + subject + 'after_frame_5_b'+ str(bSize) +'.h5')

# model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# model.summary()
# # plot_model(model, to_file='model.png', show_shapes=True, dpi=192)
# #
# # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
# # model.compile(loss='sparse_categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adadelta(), metrics=['sparse_categorical_accuracy'])
# # model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['categorical_accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=['categorical_accuracy'])

# # history = model.fit(x_train, y_train, batch_size=64, epochs=10)
# # test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=64)

# history = model.fit(x_train, y_train_C, batch_size=32, epochs=60)
# print('Testing...')
# test_loss, test_acc = model.evaluate(x_test, y_test_C, batch_size=32)

# model.save('cross_with_s5_after_frame_30.h5')
