import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential

import os
import csv

frame20 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.']
frame30 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.', 'frame_21.', 'frame_22.', 'frame_23.', 'frame_24.', 'frame_25.', 'frame_26.', 'frame_27.', 'frame_28.', 'frame_29.', 'frame_30.']
frame10 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.']

actions = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']

# model = tf.keras.models.load_model('s4-10_P006_v4_balanced.h5')

# old_model = tf.keras.models.load_model('EANN_DATA/cross_with_s4_after_frame_20.h5')
# model = Sequential()
# for layer in old_model.layers:
#     model.add(layer)

directory = os.listdir('Data/NPZ')

x_test = np.empty((1, 180, 180, 3))

corrSeq = 0
allSeq = 0

subject = 's1_'
# model = tf.keras.models.load_model('MOB_with_' + subject + 'after_frame_5_b32.h5')
model = tf.keras.models.load_model('MOB0.5_with_' + subject + 'after_frame_5_b32.h5')
# model = tf.keras.models.load_model('RES_with_' + subject + 'after_frame_5_b32.h5')

for sdir in directory:
    sdirSTR = 'Data/NPZ/' + sdir
    subdir = os.listdir(sdirSTR)
    if subject in sdirSTR:
        votes = np.zeros(10, dtype=int)
        corr = 0
        count = 0
        print('Action ' + sdirSTR + ' ------')
        # with open("MOB_" + sdir + "_votes.csv", "w+") as my_csv:
        #     csvWriter = csv.writer(my_csv, delimiter=';')
        #     with open("MOB_" + sdir + "_probs.csv", "w+") as my_csv_2:
        #         csvWriter2 = csv.writer(my_csv_2, delimiter=';')
        #         with open("MOB_" + sdir + "_preds.csv", "w+") as my_csv_3:
        #             csvWriter3 = csv.writer(my_csv_3, delimiter=';')
        #             csvWriter.writerow(actions)
        #             csvWriter2.writerow(actions)
        #             csvWriter3.writerow(actions)
        for fIdx in range(len(subdir) - 2):
            # if sum(x in subdir[fIdx] for x in frame30) == 0:
            data1 = np.load(sdirSTR + '/' + subdir[fIdx])
            data2 = np.load(sdirSTR + '/' + subdir[fIdx + 1])
            data3 = np.load(sdirSTR + '/' + subdir[fIdx + 2])
            x_test[0, :, :, :] = np.dstack((data1['sino'], data2['sino'], data3['sino']))
            prediction = model.predict(x_test)
            print('--- Found: ' + str(np.argmax(prediction)))
            corr = data1['clNum'] - 1
            print('--- Correct: ' + str(corr))
            count = count + 1
            votes[np.argmax(prediction)] = votes[np.argmax(prediction)] + 1
            probabilities = votes / count
            print(votes)
            print(probabilities)
            print(prediction[0])
            # csvWriter.writerow(votes)
            # csvWriter2.writerow(probabilities)
            # csvWriter3.writerow(prediction[0])
        # input("Next sequence...?")
        if np.argmax(votes) == corr:
            corrSeq = corrSeq + 1
        allSeq = allSeq + 1
print('Final ACC: ' + str(corrSeq/allSeq))
