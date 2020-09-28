import numpy as np

dirpath = 'NTURGB-D_120_Code/Python/raw_npy/S001C001P001R001A002.skeleton.npy'

print(int(dirpath.split("A", 1)[1][:3]))

data = np.load(dirpath,allow_pickle=True).item()
print(data['skel_body0'].shape)