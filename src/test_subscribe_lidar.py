#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import scipy.misc
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
from threading import Thread, Event
import cv2
global saved
saved = 0

def scale_to_255(a, min, max, dtype=np.uint8):
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-50, 50),
                           fwd_range = (-50, 50),
                           height_range=(-2., 2.),
                           ):
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)

    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    pixel_values = scale_to_255(pixel_values,
                                    min=height_range[0],
                                    max=height_range[1])

    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    im[y_img, x_img] = pixel_values

    return im
def lidar_bev(points):
    pointcloud = points
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
    # print(x_img.min(), x_img.max(), y_img.min(), x_img.max())

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
    print(im.shape)
    # while True:
    #     cv2.imshow('test-img', im)
    #     print(im.shape)
    #     time.sleep(1)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break

    # imshow （彩色）
    # plt.imshow(im, cmap="nipy_spectral", vmin=0, vmax=255,)
    # plt.title(path[-10:])
    # plt.xlabel('1 pixel = {} m'.format(res));plt.ylabel('1 pixel = {} m'.format(res))
    # plt.show()
from test_generate_bev import get_bev
# def
def callback(lidar):
    global saved
    if saved == 0:
        lidar = pc2.read_points(lidar)
        points = np.array(list(lidar))
        # lidar_bev(points)
        img_bevs = get_bev(points)
        time.sleep(1)
        # im = point_cloud_2_birdseye(points)
        # scipy.misc.imsave('/home/richard/projects/lidar_ws_python/src/cloud_subscribe/save/lidar.png', im)
        # saved = 1

def cloud_subscribe():
    rospy.init_node('cam_listener', disable_signals=True)
    # Thread(target=lambda: rospy.init_node('cam_listener', disable_signals=True)).start()
    # rospy.init_node('cloud_subscribe_node',anonymous=False, log_level=rospy.INFO, disable_signals=True)
    rospy.Subscriber("/point_raw", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    cloud_subscribe()
