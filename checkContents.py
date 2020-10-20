import os

directory = os.listdir('NTURGB-D_120_Code/Python/raw_npy')
minNum = 200
maxNum = 0
sumNum = 0
counter = 0

for sdir in directory:

    sdirSTR = 'NTURGB-D_120_Code/Python/raw_npy/' + sdir
    if '_NPZ' in sdirSTR:
        subdir = os.listdir(sdirSTR)
        num = len(subdir)
        if num < minNum:
            minNum = num
        if num > maxNum:
            maxNum = num
        sumNum = sumNum + num
        counter = counter + 1

print('Minimum number of frames: ', minNum)
print('Maximum number of frames: ', maxNum)
print('Average number of frames: ', sumNum/counter)
