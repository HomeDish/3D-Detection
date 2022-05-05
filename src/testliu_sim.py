"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
"""

import argparse
import sys
import os
import time

import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2,Image
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import scipy.misc
sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format
from threading import Thread
import socket
import os
import subprocess
# from my_msgs import MyImage
# from PIL import Image

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default='../checkpoints/complexer_yolo/Model_complexer_yolo_epoch_50.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.1,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',default=True,
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')
    # adding
    # configs.save_test_output = True
    # configs.saved_fn = 'video'
    # configs.output_format = 'video'

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)
    return configs
def sen_data(out_img):
    # print('queue lenth :',queue.qsize())
    # if queue.qsize()>0:
    #     out_img = queue.get()
    #     plt.imshow(out_img)
        # cv2.imshow('test',out_img)
        # cv2.waitKey(0)
    # if not os.path.exists('detect_pics/00001.png'):
    #     im2 = Image.fromarray(out_img)
    #     im2.show()
    #     # im2.save('detect_pics/00001.png')
    #     cv2.imwrite('detect_pics/00001.png',out_img)
    # print('saved!',out_img.shape)
    # time.sleep(1)
    # global out_img
    # cv2.imshow('test',out_img)
    # print('1')
    # cv2.waitKey(0)
    # plt.imshow(out_img);plt.show()

    # while True:
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break
    # im2 = Image.fromarray(out_img)
    # im2.show()


    # host = '127.0.0.1'
    # port = 12345
    # obj = socket.socket()
    # obj.connect((host, port))
    # cmd = 'netstat -ano | findstr %s' % port
    # # obj.send(bytes('this is data !!','utf-8'))
    # # put_img_pub = rospy.Publisher('/out_img', PointCloud2, queue_size=10)

    # out_data = str(out_img.reshape(-1).tolist())
    # obj.send(bytes(out_data))
    # print('send data: ',len(out_data))
    # print(np.array([float(i) for i in out_data.split(' ')]).shape)

    image_temp = Image()
    header = Header(stamp=rospy.Time.now())
    header.frame_id = 'map'
    image_temp.height = 608
    image_temp.width = 608
    image_temp.encoding = 'rgb8'
    image_temp.data = out_img.tostring()
    image_temp.header = header
    put_img_pub.publish(image_temp)
    print('publish data')
def bev_detect(imgs_bev):
    for batch_idx in range(1):
        # print(batch_idx, img_paths, imgs_bev)
        # print('imgs_bev type:',type(imgs_bev))

        # print(img_paths)
        # imgs_bev = get_bev()
        input_imgs = imgs_bev.to(device=configs.device).float()

        t1 = time_synchronized()
        outputs = model(input_imgs)
        t2 = time_synchronized()
        detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        img_bev = imgs_bev.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))

        # time.sleep(3)
        for detections in img_detections:
            if detections is None:
                continue
            # Rescale boxes to original image
            detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
            for x, y, w, l, im, re, *_, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

        # img_rgb = cv2.imread(img_paths[0])
        # calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
        # objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
        # img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

        img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)
        # adding
        # global out_img
        out_img = img_bev
        print('generate data')
        # queue.put(out_img)
        time.sleep(0.1)
        sen_data(out_img)
        # out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=608)

        # print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
        #                                                                                1 / (t2 - t1)))
        #
        # if configs.save_test_output:
        #     if configs.output_format == 'image':
        #         # img_fn = os.path.basename(img_paths[0])[:-4]
        #         # cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
        #         pass
        #     elif configs.output_format == 'video':
        #         if out_cap is None:
        #             out_cap_h, out_cap_w = out_img.shape[:2]
        #             fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #             out_cap = cv2.VideoWriter(
        #                 os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
        #                 fourcc, 30, (out_cap_w, out_cap_h))
        #
        #         out_cap.write(out_img)
        #     else:
        #         raise TypeError
        #
        # if configs.show_image:
        #     # Thread(target=lambda: cv2.imshow('test-img', out_img)).start()
        #     # cv2.imshow('test-img', out_img)
        #     print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
        #
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         cv2.destroyAllWindows()
        #         break
            # if cv2.waitKey(0) & 0xFF == 27:
            #     break

def callback(lidar):
    lidar = pc2.read_points(lidar)
    points = np.array(list(lidar))
    imgs_bev = get_bev(points)
    bev_detect(imgs_bev)
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()
def cloud_subscribe():
    # rospy.init_node('process_node', disable_signals=True)
    # Thread(target=lambda: rospy.init_node('cam_listener', disable_signals=True)).start()
    # rospy.init_node('cloud_subscribe_node',anonymous=False, log_level=rospy.INFO, disable_signals=True)

    rospy.Subscriber("/point_raw", PointCloud2, callback)
    rospy.spin()


if __name__ == '__main__':

    configs = parse_test_configs()
    # print(configs)
    configs.distributed = False  # For testing

    model = create_model(configs)

    # model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    print(configs.device)
    out_cap = None
    model.eval()
    # test_dataloader = create_test_dataloader(configs)
    rospy.init_node('process_node', disable_signals=True)
    put_img_pub = rospy.Publisher('/out_img', Image, queue_size=10)
    from test_generate_bev import get_bev

    ###
    out_img = np.array((608,608,3))
    from queue import Queue
    # queue = Queue()
    # t1 = Thread(target=cloud_subscribe)
    # t2 = Thread(target=sen_data)
    with torch.no_grad():
        # for batch_idx, (img_paths, imgs_bev) in enumerate(test_dataloader):
        cloud_subscribe() # subscribe detect
        # bev_detect(imgs_bev=get_bev()) ## for debug

        # t2.start()
        # t1.start()
        # t1.join()
        # t2.join()
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
