import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Dropout

import random
import os

old_model = tf.keras.models.load_model('EANN_DATA/cross_with_s4_after_frame_20.h5')

old_model.summary()

new_model = Sequential()

for layer in old_model.layers[:-4]:
    new_model.add(layer)
    # if 'conv' in layer.name:
    #     new_model.add(BatchNormalization())

new_model.trainable = False

for layer in new_model.layers:
    if 'batch_norm' in layer.name:
        layer.trainable = True

new_model.add(Flatten())
new_model.add(Dense(256, activation='relu'))
new_model.add(Dropout(0.5, name='dropout_3'))
new_model.add(Dense(6, activation='softmax'))

new_model.summary()