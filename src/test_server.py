#! python3
# SocketTestv0.1.py
# -*- coding:utf-8 -*-


import socket
import os
import time
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
def get_data(conn):
    ret = conn.recv(102400).decode('utf-8')
    ret = np.array(ret)
    print(ret.shape)
    # while True:
    #
    #     # print(conn)
    #     ret = conn.recv(102400).decode('utf-8')
    #     ret = np.array(ret)
    #     print(ret.shape)
    #     continue
        # print('get 1 messages',ret);break
        # if 'End' in ret:
        #     print('End')
        #     conn.sendall(bytes("Finish!", encoding="utf-8"))
        #     # conn.close()#sk.close 不发送fin包
        #     os.system('netstat -ano | findstr %s' % port)
        #     break
        # else:
        #     print(np.array(ret))
        #     print('Not close')
        #     continue


def callback(lidar):
    lidar = pc2.read_points(lidar)
    points = np.array(list(lidar))
    print(points.shape)


def cloud_subscribe():
    rospy.init_node('out_listener', disable_signals=True)
    # Thread(target=lambda: rospy.init_node('cam_listener', disable_signals=True)).start()
    # rospy.init_node('cloud_subscribe_node',anonymous=False, log_level=rospy.INFO, disable_signals=True)

    rospy.Subscriber("/out_img", PointCloud2, callback)
    rospy.spin()
if __name__ == '__main__':
    # cloud_subscribe()
    host = '127.0.0.1'
    port = 12345
    sk = socket.socket()
    sk.bind((host, port))
    sk.listen(5)
    print('The port is listening...')
    os.system('netstat -ano | findstr %s' % port)
    print('Wait for the client.')
    while True:
        # sk.settimeout(10)
        conn, address = sk.accept()
        print('Connecting...')
        print('Connect from: ', address)
        # os.system('netstat -ano | findstr %s' % port)
        ret=conn.recv(1024000)
    #     # ret = np.array([float(i) for i in conn.recv(1024000).decode('utf-8').split(' ')])
        if ret:
            print(type(ret))
            print(np.array(list(ret.decode('utf-8'))).shape)
    #     # ret = np.array(ret)
    #     # print(ret.shape)
        # get_data(conn)