import numpy as np
import tensorflow as tf

import random
import os

import time

# ALLITEMS = 5656

x_train_L = []
y_train = []
x_test_L = []
y_test = []

frame20 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.']
frame30 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.', 'frame_10.', 'frame_11.', 'frame_12.', 'frame_13.', 'frame_14.', 'frame_15.', 'frame_16.', 'frame_17.', 'frame_18.', 'frame_19.', 'frame_20.', 'frame_21.', 'frame_22.', 'frame_23.', 'frame_24.', 'frame_25.', 'frame_26.', 'frame_27.', 'frame_28.', 'frame_29.', 'frame_30.']
frame10 = ['frame_5.', 'frame_6.', 'frame_7.', 'frame_8.', 'frame_9.']

actionLists = [['A023'], ['A007'], ['A094'], ['A012'], ['A050'], ['A010']]
separator = ', '
oldActIdxs = [[0, 6], [2], [2], [3, 4, 5], [7], [9]]

# actionLists = [['A023'], ['A007'], ['A050'], ['A010']]
# separator = ', '
# oldActIdxs = [[0, 6], [2], [7], [9]]

# testSubjects = ['P001', 'P043']

directory = os.listdir('NTURGB-D_120_Code/Python/raw_npy')
ALLITEMS = len(directory)

model = tf.keras.models.load_model('EANN_DATA/cross_with_s4_after_frame_20.h5')

# f1 = open('samples_v4_6cl_10fr_multi_noRand.txt', "a")
# f2 = open('P001_P043_testing.txt', "a")
f3 = open('average_times_infer.txt', "a")
avgTimeInfer = 0
framecount = 0

for idx in range(len(actionLists)):
    x_test = np.empty((1, 180, 180, 1))

    corrSeq = 0
    allSeq = 0

    counter = 0

    for sdir in directory:

        counter = counter + 1

        typeStr = separator.join(actionLists[idx]) + ': reading ' + str(counter) + '/' + str(ALLITEMS) + '...'
        print(typeStr, end="\r")

        sdirSTR = 'NTURGB-D_120_Code/Python/raw_npy/' + sdir
        if '_NPZ' in sdirSTR and sum(x in sdirSTR for x in actionLists[idx]) != 0:

            votes = np.zeros(10, dtype=int)
            # corr = 0
            count = 0
            # print('Action ' + sdirSTR + ' ------')
            subdir = os.listdir(sdirSTR)

            # randClass = random.choice(oldActIdxs[idx])

            sumTimeVid = 0
            frameCountVid = 0

            for fname in subdir:
                if sum(x in fname for x in frame10) == 0:
                    data = np.load(sdirSTR + '/' + fname)
                    start = time.time()
                    x_test[0, :, :, 0] = data['sino']
                    prediction = model.predict(x_test)
                    # print('--- Found: ' + str(np.argmax(prediction)))
                    # corr = data['clNum'] - 1
                    # print('--- Correct: ' + str(corr))
                    count = count + 1
                    votes[np.argmax(prediction)] = votes[np.argmax(prediction)] + 1
                    probabilities = votes / count
                    end = time.time()
                    avgTimeInfer = avgTimeInfer + (end - start)
                    sumTimeVid = sumTimeVid + (end - start)
                    framecount = framecount + 1
                    frameCountVid = frameCountVid + 1
                    # print(votes)
                    # print(probabilities)
                    # print(prediction[0])
            f3.write('********************************************************\n')
            f3.write('SPECIFIC video avg time per frame: {}\n'.format(str(sumTimeVid / frameCountVid)))
            f3.write('Average time for inference per frame: {}\n'.format(str(avgTimeInfer / framecount)))
            f3.flush()
            # input("Next sequence...?")
            if np.argmax(votes) not in oldActIdxs[idx]:
                maxVotes = 0
                maxIndex = 0
                randClass = 0
                for idx2 in oldActIdxs[idx]:
                    if votes[idx2] > maxVotes:
                        maxVotes = votes[idx2]
                        maxIndex = idx2
                if maxVotes == 0:
                    randClass = random.choice(oldActIdxs[idx])
                else:
                    randClass = maxIndex
                # f1.write('{0};{1}\n'.format(sdir, randClass))
                # f1.flush()

# f1.close()
f3.close()
