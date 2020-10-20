import numpy as np

import tensorflow as tf
import os
import csv

import sys

frame20 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.']
frame30 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.', 'frame_21.', 'frame_22.', 'frame_23.', 'frame_24.', 'frame_25.', 'frame_26.', 'frame_27.', 'frame_28.', 'frame_29.', 'frame_30.']

actions = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']

# models = ['cross_with_s6_after_frame_20.h5', 'cross_with_s6_after_frame_30.h5', 'cross_with_s4_after_frame_20.h5',
#           'cross_with_s4_after_frame_30.h5', 'cross_with_s5_after_frame_20.h5', 'cross_with_s5_after_frame_30.h5']

# models = ['cross_with_s4_after_frame_20.h5', 's4-20_P001-P043.h5']

models = ['cross_with_s4_after_frame_20.h5']

# actionLists = [['A023'], ['A007'], ['A094'], ['A007', 'A094'], ['A012'], ['A050'], ['A010']]
#
# oldActIdxs = [[0, 6], [2], [2], [2], [3, 4, 5], [7], [9]]

actionLists = ['A023', 'A007', 'A050', 'A010']

oldActIdxs = [[0, 6], [2], [7], [9]]

# print('Action list: ', actionLists[3])
# print(len(actionLists))
# print(len(oldActIdxs))

directory = os.listdir('NTURGB-D_120_Code/Python/raw_npy')

f = open('outputLog_20_2.txt', "a")

# f.write('Action list: {}\n'.format(actionLists[3]))
# f.write('Old indexes: {}'.format(oldActIdxs[0]))

for fileName in models:

    model = tf.keras.models.load_model('EANN_DATA/' + fileName)

    f.write('*******************************************\n')
    f.write('Model: {}\n'.format(fileName))
    f.flush()

    corrSeqM = 0
    allSeqM = 0

    for idx in range(len(actionLists)):

        x_test = np.empty((1, 180, 180, 1))

        corrSeq = 0
        allSeq = 0

        counter = 0

        for sdir in directory:

            counter = counter + 1

            typeStr = fileName + ' - ' + actionLists[idx] + ': testing ' + str(counter) + '/' + str(len(directory)) + '...'
            print(typeStr, end="\r")

            sdirSTR = 'NTURGB-D_120_Code/Python/raw_npy/' + sdir
            if '_NPZ' in sdirSTR and sum(x in sdirSTR for x in actionLists[idx]) != 0:
                votes = np.zeros(10, dtype=int)
                # corr = 0
                count = 0
                # print('Action ' + sdirSTR + ' ------')
                subdir = os.listdir(sdirSTR)
                for fname in subdir:
                    if sum(x in fname for x in frame20) == 0:
                        data = np.load(sdirSTR + '/' + fname)
                        x_test[0, :, :, 0] = data['sino']
                        prediction = model.predict(x_test)
                        # print('--- Found: ' + str(np.argmax(prediction)))
                        # corr = data['clNum'] - 1
                        # print('--- Correct: ' + str(corr))
                        count = count + 1
                        votes[np.argmax(prediction)] = votes[np.argmax(prediction)] + 1
                        probabilities = votes / count
                        # print(votes)
                        # print(probabilities)
                        # print(prediction[0])
                # input("Next sequence...?")
                if np.argmax(votes) in oldActIdxs[idx]:
                    corrSeq = corrSeq + 1
                    corrSeqM = corrSeqM + 1
                allSeq = allSeq + 1
                allSeqM = allSeqM + 1
        f.write('-------------------------------------------\n')
        f.write('Action list: {}\n'.format(actionLists[idx]))
        f.write('Old indexes: {}\n'.format(oldActIdxs[idx]))
        f.write('FINAL ACC: {}\n'.format(str(corrSeq/allSeq)))
        f.flush()

    f.write('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n')
    f.write('MODEL ACC: {}\n'.format(str(corrSeqM / allSeqM)))
    f.flush()

f.close()
