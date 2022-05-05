import os
import sys
import numpy as np
import pandas as pd
txt_path = r'/home/bdsc/Desktop/Lidar/record/raw_data'
csv_path = r'/home/bdsc/Desktop/Lidar/record/testing/velodyne'
files = sorted(os.listdir(txt_path))
# for file in os.listdir(txt_path):
#     # if os.path.splitext(file)[1]=='.TXT':
#     files.append(file)
for idx,file in enumerate(files):
    fsize = os.path.getsize(os.path.join(txt_path, file))
    if fsize<1000000:
        os.remove(os.path.join(txt_path, file))
        files.remove(file)
for idx,file in enumerate(files):
    print(file)
    Frame = pd.read_csv(os.path.join(txt_path, file), header=None,sep=' ',dtype=np.float32,engine='python')

        # new_data = np.array([eval(item) for item in data])
        #
        # print(new_data[:5])
    # print(Frame.head())
    Frame.to_csv(os.path.join(csv_path,'{:06d}.csv'.format(idx)),header=None,index=False)
    if idx == 0:
        data = pd.read_csv(os.path.join(csv_path,'{:06d}.csv'.format(idx)),header=None)
        data = np.array(data).reshape(-1,4).astype(np.float32)
        # print(Frame.head())
        # data = np.array(Frame)
        print(data.shape)
        print(type(data[1,1]))
        print(data[:5])
        # print(type(Frame.iloc[1, 1]))
    # for idx, file in enumerate(files):
    #     data = np.fromfile(os.path.join(txt_path,file),dtype=np.float32)
    #     print(data.shape)
        # Frame = pd.read_csv(os.path.join(txt_path,file),header=None)
        # print(Frame.head())
        # Frame.to_pickle(os.path.join(txt_path,'testing/velodyne','{:06d}.bin'.format(idx)))
# path = os.path.join(txt_path,'testing/velodyne','000001.bin')
# data = np.fromfile(path).reshape(-1,4)
# print(data[:4,:])