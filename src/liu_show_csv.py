import os
import pandas as pd
import time
import sys
sys.path.append('../')
# print(os.getcwd())
from data_process import kitti_data_utils, kitti_bev_utils,kitti_dataset
from data_process.kitti_dataloader import create_test_dataloader
#binary
import numpy as np
import mayavi.mlab
import matplotlib.pyplot as plt
from PIL import Image

# lidar_path换成自己的.bin文件路径
def read_csv(path):
    if path[-4:] == '.bin':
        pointcloud = np.fromfile(path, dtype=np.float32, count=-1).reshape(-1, 4)
    else:
        data = pd.read_csv(path, header=None)
        pointcloud = np.array(data).reshape(-1, 4).astype(np.float32)
    print(pointcloud.shape)
    pointcloud = pointcloud
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point

    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

    degr = np.degrees(np.arctan(z / d))

    vals = 'height'
    if vals == "height":
        col = z
    else:
        col = d

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )

    mayavi.mlab.show()
    # time.sleep(1)
def csv_bev(path):
    # import numpy as np
    # 点云读取
    # pointcloud = np.fromfile(str("000010.bin"), dtype=np.float32, count=-1).reshape([-1, 4])
    if path[-4:] == '.bin':
        pointcloud = np.fromfile(path, dtype=np.float32, count=-1).reshape(-1, 4)
    else:
        data = pd.read_csv(path, header=None)
        # print('describe:',data.describe())
        pointcloud = np.array(data).reshape(-1, 4).astype(np.float32)
    # print("shape:",pointcloud.shape)
    # print("min:",np.min(pointcloud,axis=0))
    # print("max:",np.max(pointcloud,axis=0))
    # 设置鸟瞰图范围
    # side_range = (-40, 40)  # 左右距离
    # fwd_range = (0, 70.4)  # 后前距离
    side_range = (-50, 50)  # 左右距离
    fwd_range = (-100, 0)  # 后前距离

    x_points = pointcloud[:, 0]
    y_points = pointcloud[:, 1]
    z_points = pointcloud[:, 2]

    # 获得区域内的点
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    res = 0.1  # 分辨率0.05m
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # 调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    print(x_img.min(), x_img.max(), y_img.min(), x_img.max())

    # 填充像素值
    height_range = (-2, 0.5)
    pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])

    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)

    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])

    # 创建图像数组
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value

    # imshow （灰度）
    # im2 = Image.fromarray(im)
    # im2.show()

    # imshow （彩色）
    plt.imshow(im, cmap="nipy_spectral", vmin=0, vmax=255,)
    plt.title(path[-10:])
    plt.xlabel('1 pixel = {} m'.format(res));plt.ylabel('1 pixel = {} m'.format(res))
    # plt.show()

#mayavi显示点云
csv_path = r'/home/bdsc/Desktop/Lidar/Complex-YOLOv4-Pytorch/dataset/kitti/testing/velodyne' ## C32 data
# csv_path = r'../dataset/kitti/training_kitti/velodyne' ## KITTI
# csv_path = r'../../record/data4'
files = sorted(os.listdir(csv_path))
for file in files:
    data_path = os.path.join(csv_path,file)
    # print(data_path)
    # read_csv(data_path) ## 3d
    csv_bev(data_path);plt.pause(0.5) ## 2d

