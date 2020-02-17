import numpy as np

import tensorflow as tf
import os

model = tf.keras.models.load_model('s2_to_s6_model.h5')

directory = os.listdir('Data/NPZ')

x_test = np.empty((1, 180, 180, 1))
for sdir in directory:
    sdirSTR = 'Data/NPZ/' + sdir
    subdir = os.listdir(sdirSTR)
    if 's1_' in sdirSTR:
        print('Action ' + sdirSTR + ' ------')
        for fname in subdir:
            print('-' + fname)
            data = np.load(sdirSTR + '/' + fname)
            x_test[0, :, :, 0] = data['sino']
            prediction = model.predict(x_test)
            print('--- Found: ' + str(np.argmax(prediction)))
            corr = data['clNum'] - 1
            print('--- Correct: ' + str(corr))
