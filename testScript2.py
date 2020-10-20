import numpy as np

import random
import os

file1 = open('samples_v4_6cl_10fr_multi_noRand.txt', 'r')
dataLines = file1.readlines()

classDict = {6 : 0}

# counter = 0

for line in dataLines:

    # counter = counter + 1
    # typeStr = 'Tested ' + str(counter) + '/' + str(len(dataLines)) + ' lines...'
    # print(typeStr, end="\r")

    tokenized = line.split(';')
    # sdir = tokenized[0]
    classIdx = int(tokenized[1])
    if classIdx in classDict:
        classDict[classIdx] = classDict[classIdx] + 1
    else:
        classDict[classIdx] = 1

for key in classDict:
    # print(key, ' : ', personDict[key], ' - Videos: ', numDict[key])
    print(key, ' : ', classDict[key])
