import sys
import os
import random
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from data_process import transformation, kitti_bev_utils, kitti_data_utils
import config.kitti_config as cnf
sys.path.append('../')
def get_bev(lidarData=np.array(pd.read_csv('../dataset/kitti/testing/velodyne/000000.csv', header=None)).reshape(-1, 4).astype(np.float32)):
    # lidarData = np.array(pd.read_csv('../dataset/kitti/testing/velodyne/000000.csv', header=None)).reshape(-1, 4).astype(np.float32)
    lidarData = lidarData.reshape(-1, 4).astype(np.float32)
    b = kitti_bev_utils.removePoints(lidarData, cnf.boundary)
    rgb_map = kitti_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
    rgb_map = np.array([rgb_map])
    return torch.from_numpy(rgb_map)


# import argparse
# import sys
# import os
# import time
#
# import matplotlib.pyplot as plt
# from easydict import EasyDict as edict
# import cv2
# import torch
# import numpy as np
#
# sys.path.append('../')
#
# import config.kitti_config as cnf
# from data_process import kitti_data_utils, kitti_bev_utils
# from data_process.kitti_dataloader import create_test_dataloader
# from models.model_utils import create_model
# from utils.misc import make_folder
# from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
# from utils.misc import time_synchronized
# from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format
# def parse_test_configs():
#     parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
#     parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
#                         help='The name using for saving logs, models,...')
#     parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
#                         help='The name of the model architecture')
#     parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
#                         help='The path for cfgfile (only for darknet)')
#     parser.add_argument('--pretrained_path', type=str, default='../checkpoints/complexer_yolo/Model_complexer_yolo_epoch_50.pth', metavar='PATH',
#                         help='the path of the pretrained checkpoint')
#     parser.add_argument('--use_giou_loss', action='store_true',
#                         help='If true, use GIoU loss during training. If false, use MSE loss for training')
#
#     parser.add_argument('--no_cuda', action='store_true',
#                         help='If true, cuda is not used.')
#     parser.add_argument('--gpu_idx', default=0, type=int,
#                         help='GPU index to use.')
#
#     parser.add_argument('--img_size', type=int, default=608,
#                         help='the size of input image')
#     parser.add_argument('--num_samples', type=int, default=None,
#                         help='Take a subset of the dataset to run and debug')
#     parser.add_argument('--num_workers', type=int, default=1,
#                         help='Number of threads for loading data')
#     parser.add_argument('--batch_size', type=int, default=1,
#                         help='mini-batch size (default: 4)')
#
#     parser.add_argument('--conf_thresh', type=float, default=0.5,
#                         help='the threshold for conf')
#     parser.add_argument('--nms_thresh', type=float, default=0.1,
#                         help='the threshold for conf')
#
#     parser.add_argument('--show_image', action='store_true',default=True,
#                         help='If true, show the image during demostration')
#     parser.add_argument('--save_test_output', action='store_true',
#                         help='If true, the output image of the testing phase will be saved')
#     parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
#                         help='the type of the test output (support image or video)')
#     parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
#                         help='the video filename if the output format is video')
#
#     configs = edict(vars(parser.parse_args()))
#     configs.pin_memory = True
#
#     ####################################################################
#     ##############Dataset, Checkpoints, and results dir configs#########
#     ####################################################################
#     configs.working_dir = '../'
#     configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')
#     # adding
#     configs.save_test_output = True
#     configs.saved_fn = 'video'
#     configs.output_format = 'video'
#
#     if configs.save_test_output:
#         configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
#         make_folder(configs.results_dir)
#     return configs
# configs = parse_test_configs()
#     # print(configs)
# configs.distributed = False  # For testing
#
# model = create_model(configs)
#     # model.print_network()
# print('\n\n' + '-*=' * 30 + '\n\n')
# assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
# model.load_state_dict(torch.load(configs.pretrained_path))
#
# configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
# model = model.to(device=configs.device)
# out_cap = None
# model.eval()

# imgs_bev = torch.from_numpy(rgb_map)
# img_bev = imgs_bev.squeeze() * 255
# img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
# img_bev = cv2.resize(img_bev, (608, 608))

