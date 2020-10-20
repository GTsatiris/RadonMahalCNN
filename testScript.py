import numpy as np
import tensorflow as tf

import random
import os

directory = os.listdir('NTURGB-D_120_Code/Python/raw_npy')
personDict = {"001": []}
numDict = {"001": 0}
setupDict = {"001": []}
activityDict = {"023": []}

for sdir in directory:
    if '_NPZ' in sdir:
        # person = sdir.split("P", 1)[1][:3]
        # if person in personDict:
        #     numDict[person] = numDict[person] + 1
        #     activity = sdir.split("A", 1)[1][:3]
        #     setup = sdir.split("S", 1)[1][:3]
        #     if activity not in personDict[person]:
        #         personDict[person].append(activity)
        #     if setup not in setupDict[person]:
        #         setupDict[person].append(setup)
        # else:
        #     numDict[person] = 1
        #     personDict[person] = []
        #     setupDict[person] = []
        #     activity = sdir.split("A", 1)[1][:3]
        #     setup = sdir.split("S", 1)[1][:3]
        #     if activity not in personDict[person]:
        #         personDict[person].append(activity)
        #     if setup not in setupDict[person]:
        #         setupDict[person].append(setup)
        activity = sdir.split("A", 1)[1][:3]
        if activity in activityDict:
            setup = sdir.split("S", 1)[1][:3]
            if setup not in activityDict[activity]:
                activityDict[activity].append(setup)
        else:
            activityDict[activity] = []
            setup = sdir.split("S", 1)[1][:3]
            if setup not in activityDict[activity]:
                activityDict[activity].append(setup)

for key in activityDict:
    # print(key, ' : ', personDict[key], ' - Videos: ', numDict[key])
    print(key, ' : ', activityDict[key])
