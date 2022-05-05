
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../')
sys.path.append('../src')
sys.path.append('../src/config')

from data_process.kitti_dataset import KittiDataset
from data_process.transformation import Compose, OneOf, Random_Rotation, Random_Scaling, Horizontal_Flip, Cutout


def create_train_dataloader(configs):
    """Create dataloader for training"""

    train_lidar_transforms = OneOf([
        Random_Rotation(limit_angle=20., p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0)
    ], p=0.66)

    train_aug_transforms = Compose([
        Horizontal_Flip(p=configs.hflip_prob),
        Cutout(n_holes=configs.cutout_nholes, ratio=configs.cutout_ratio, fill_value=configs.cutout_fill_value,
               p=configs.cutout_prob)
    ], p=1.)

    train_dataset = KittiDataset(configs.dataset_dir, mode='train', lidar_transforms=train_lidar_transforms,
                                 aug_transforms=train_aug_transforms, multiscale=configs.multiscale_training,
                                 num_samples=configs.num_samples, mosaic=configs.mosaic,
                                 random_padding=configs.random_padding)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler,
                                  collate_fn=train_dataset.collate_fn)

    return train_dataloader, train_sampler


def create_val_dataloader(configs):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = KittiDataset(configs.dataset_dir, mode='val', lidar_transforms=None, aug_transforms=None,
                               multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False)
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler,
                                collate_fn=val_dataset.collate_fn)

    return val_dataloader


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_dataset = KittiDataset(configs.dataset_dir, mode='test', lidar_transforms=None, aug_transforms=None,
                                multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader


if __name__ == '__main__':
    import argparse
    import os

    import cv2
    import numpy as np
    from easydict import EasyDict as edict

    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target
    import config.kitti_config as cnf

    parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--hflip_prob', type=float, default=0.,
                        help='The probability of horizontal flip')
    parser.add_argument('--cutout_prob', type=float, default=0.,
                        help='The probability of cutout augmentation')
    parser.add_argument('--cutout_nholes', type=int, default=1,
                        help='The number of cutout area')
    parser.add_argument('--cutout_ratio', type=float, default=0.3,
                        help='The max ratio of the cutout area')
    parser.add_argument('--cutout_fill_value', type=float, default=0.,
                        help='The fill value in the cut out area, default 0. (black)')
    parser.add_argument('--multiscale_training', action='store_true',
                        help='If true, use scaling data for training')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--mosaic', action='store_true',
                        help='If true, compose training samples as mosaics')
    parser.add_argument('--random-padding', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--show-train-data', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')
    parser.add_argument('--save_img', action='store_true',
                        help='If true, save the images')

    configs = edict(vars(parser.parse_args()))
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.dataset_dir = os.path.join('../', 'dataset', 'kitti')

    if configs.save_img:
        print('saving validation images')
        configs.saved_dir = os.path.join(configs.dataset_dir, 'validation_data')
        if not os.path.isdir(configs.saved_dir):
            os.makedirs(configs.saved_dir)

    if configs.show_train_data:
        dataloader, _ = create_train_dataloader(configs)
        print('len train dataloader: {}'.format(len(dataloader)))
    else:
        dataloader = create_val_dataloader(configs)
        print('len val dataloader: {}'.format(len(dataloader)))

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')
    ### add by liujialai --2021-12-03
    targets_no_box = []
    ### end add

    for batch_i, (img_files, imgs, targets) in enumerate(dataloader):
        if not (configs.mosaic and configs.show_train_data):
            img_file = img_files[0]
            img_rgb = cv2.imread(img_file)
            calib = kitti_data_utils.Calibration(img_file.replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = invert_target(targets[:, 1:], calib, img_rgb.shape, RGB_Map=None)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)
            ### add by liujialai --2021-11-03
            print(img_file)
            ### end add
        ### add by liujialai --2021-12-03
        if targets.numpy().shape[0] == 0:
            targets_no_box.append('{:06d}'.format(batch_i))
            with open('delete_nobox_sample.txt', 'a+') as f:
                f.write('{:06d}\n'.format(batch_i))
        ### end add
    ### delete corresponding csv and txt
    base_dir = os.path.join('../dataset/kitti/training')
    csv_path = os.path.join(base_dir, 'velodyne')
    txt_path = os.path.join(base_dir, 'label_2')
    with open('delete_nobox_sample.txt', 'r') as f:
        for line in f.readlines():
            del_sample_id = line.strip()
            try:
                os.remove(os.path.join(csv_path, del_sample_id + '.csv'))
                os.remove(os.path.join(txt_path, del_sample_id + '.txt'))
            except:
                pass
            print('{} sample deleted !!!'.format(del_sample_id))
    # %%
    data = [line[:-4] for line in os.listdir(csv_path)]
    data.sort()
    for idx, file_id in enumerate(data):
        os.rename(os.path.join(csv_path, file_id + '.csv'), os.path.join(csv_path, '{:06d}'.format(idx) + '.csv'))
        os.rename(os.path.join(txt_path, file_id + '.txt'), os.path.join(txt_path, '{:06d}'.format(idx) + '.txt'))
        print('{} sample is renamed as {:06d} !!!'.format(file_id,idx))
    ### amend tarin.txt and val.txt in ImageSets
    LEN = len(data)
    imageset_path = '../dataset/kitti/ImageSets'
    with open(os.path.join(imageset_path, 'train.txt'), 'w') as f_train:
        for i in range(LEN):
            f_train.write('{:06d}\n'.format(i))
    with open(os.path.join(imageset_path, 'val.txt'), 'w') as f_val:
        for i in range(LEN):
            f_val.write('{:06d}\n'.format(i))
        
        



