import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import random
import os

x_train_L = []
y_train = []
x_test_L = []
y_test = []

frame20 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.']
frame30 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.', 'frame_21.', 'frame_22.', 'frame_23.', 'frame_24.', 'frame_25.', 'frame_26.', 'frame_27.', 'frame_28.', 'frame_29.', 'frame_30.']
frame10 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.']

# actionLists = [['A023'], ['A007'], ['A094'], ['A007', 'A094'], ['A012'], ['A050'], ['A010']]
# separator = ', '
# oldActIdxs = [[0, 6], [2], [2], [2], [3, 4, 5], [7], [9]]

# actionLists = [['A023'], ['A007'], ['A050'], ['A010']]
# separator = ', '
# oldActIdxs = [[0, 6], [2], [7], [9]]

testSubject = 'P006'
trainSamples = 20

directory = os.listdir('NTURGB-D_120_Code/Python/raw_npy')
ALLITEMS = len(directory)

file1 = open('samples_v4_6cl_10fr_multi_noRand.txt', 'r')
dataLines = file1.readlines()
trainLines = []
testLines = []

counter = 0

for line in dataLines:
    
    if testSubject in line:
        testLines.append(line)
    else:
        trainLines.append(line)

for line in testLines:

    counter = counter + 1
    typeStr = 'Validation set: Loaded ' + str(counter) + ' samples...'
    print(typeStr, end="\r")

    tokenized = line.split(';')
    sdir = tokenized[0]
    classIdx = int(tokenized[1])

    sdirSTR = 'NTURGB-D_120_Code/Python/raw_npy/' + sdir
    subdir = os.listdir(sdirSTR)
    for fname in subdir:
        if sum(x in fname for x in frame20) == 0:
            data = np.load(sdirSTR + '/' + fname)
            x_test_L.append(data['sino'])
            y_test.append(classIdx)

print('')
print('Validation set: DONE! Size: ', counter)
counter = 0

classDict = {6 : 0}
pathDict = {6 : []}
# counter = 0

for line in trainLines:

    # counter = counter + 1
    # typeStr = 'Tested ' + str(counter) + '/' + str(len(dataLines)) + ' lines...'
    # print(typeStr, end="\r")

    tokenized = line.split(';')
    sdir = tokenized[0]
    classIdx = int(tokenized[1])
    if classIdx in classDict:
        classDict[classIdx] = classDict[classIdx] + 1
        pathDict[classIdx].append(sdir)
    else:
        classDict[classIdx] = 1
        pathDict[classIdx] = []

print('')

for key in classDict:

    randomList = random.sample(range(0, len(pathDict[key]) - 1), trainSamples)

    for idx in range(len(randomList)):
        lineIdx = randomList[idx]
        sdir = pathDict[key][lineIdx]
        classIdx = key

        sdirSTR = 'NTURGB-D_120_Code/Python/raw_npy/' + sdir
        subdir = os.listdir(sdirSTR)
        for fname in subdir:
            if sum(x in fname for x in frame20) == 0:
                data = np.load(sdirSTR + '/' + fname)
                x_train_L.append(data['sino'])
                y_train.append(classIdx)

print('')
print('Training set: DONE!')


# randomList = random.sample(range(0, len(trainLines) - 1), trainSamples)
#
# for idx in range(len(randomList)):
#
#     lineIdx = randomList[idx]
#     line = trainLines[lineIdx]
#
#     counter = counter + 1
#     typeStr = 'Training set: Loaded ' + str(counter) + '/' + str(trainSamples) + ' samples...'
#     print(typeStr, end="\r")
#
#     tokenized = line.split(';')
#     sdir = tokenized[0]
#     classIdx = int(tokenized[1])
#
#     sdirSTR = 'NTURGB-D_120_Code/Python/raw_npy/' + sdir
#     subdir = os.listdir(sdirSTR)
#     for fname in subdir:
#         if sum(x in fname for x in frame20) == 0:
#             data = np.load(sdirSTR + '/' + fname)
#             x_train_L.append(data['sino'])
#             y_train.append(classIdx)
#
# print('Training set: DONE!')
#
model = tf.keras.models.load_model('EANN_DATA/cross_with_s4_after_frame_20.h5')

x_train = np.asarray(x_train_L).reshape([len(x_train_L), 180, 180, 1])
x_test = np.asarray(x_test_L).reshape([len(x_test_L), 180, 180, 1])
y_train_C = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_C = tf.keras.utils.to_categorical(y_test, num_classes=10)

print('Training...')

model.trainable = True
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999), metrics=['categorical_accuracy'])

history = model.fit(x_train, y_train_C, batch_size=32, epochs=80, use_multiprocessing=True)

print('Testing...')

test_loss, test_acc = model.evaluate(x_test, y_test_C, batch_size=32)

model.save('s4-10_' + testSubject + '_v4_balanced.h5')
