#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import shutil

import yaml
import argparse
import transforms3d as tfs

LABEL_DICT={
    'person':'Pedestrian',
    'car':'Car',
    'bicycle':'Cyclist',
    '':'DontCare',

}

YAML_FILE="data5.yaml"

ROOT='data/dataset'
LABEL_FOLDER=YAML_FILE[:-5]
DATASET_FOLDER=ROOT+'/ImageSets'

def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.load(file_data)
    return data

def quat2euler_z(w,x,y,z):
    angles=tfs.euler.quat2euler([w,x,y,z])
    return angles[2]

if __name__=='__main__':
    parser = argparse.ArgumentParser('Please input the annotation file path you want to transfer using --input=""')
    parser.add_argument(
        '-i',
        '--file',
        type=str,
        default=YAML_FILE,
        help='Path to the file you want to transfer.')

    args = parser.parse_args()
    if args.file is None:
        raise FileNotFoundError("Please input the file's path to start!")
    else:
        # args.input = os.Path(args.input)
        print(args.file)
        YAML_FILE = args.file
        LABEL_FOLDER = args.file[:-5]

    if not os.path.exists(LABEL_FOLDER):
        os.makedirs(LABEL_FOLDER)
    else:
        shutil.rmtree(LABEL_FOLDER)
        print('remove previous {} !!!'.format(LABEL_FOLDER))
        os.makedirs(LABEL_FOLDER)
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)


    d=get_yaml_data(yaml_file=YAML_FILE)
    labels=d['labels']
    tags=d['tags']
    tracks=d['tracks']

    for id_tracks in tracks:
        track=id_tracks['track']
        for t in track:
            name=str(t['header']['stamp']['secs'])+'.'+ '%09d' % t['header']['stamp']['nsecs']
            label=t['label']

            # x=t['translation']['x']
            # y=t['translation']['y']
            # z=t['translation']['z']
            # qw = t['rotation']['w']
            # qx = t['rotation']['x']
            # qy = t['rotation']['y']
            # qz = t['rotation']['z']
            # l = t['box']['length']
            # w = t['box']['width']
            # h = t['box']['height']
            ### remod vy liujiali --2021-11-30
            x = t['translation']['y']*(-1)
            y = t['translation']['z']*(-1)
            z = t['translation']['x']
            qw = t['rotation']['w']
            qx = t['rotation']['y']*(-1)
            qy = t['rotation']['z']*(-1)
            qz = t['rotation']['x']
            w = t['box']['length']
            l = t['box']['width']
            h = t['box']['height']
            ### end remod



            r=quat2euler_z(qw,qx,qy,qz)
            label_line=LABEL_DICT[label]+' 0.00 0 0 642.24 178.50 680.14 208.68'+' '+str(h)+' '+str(w)+' '+str(l)+' '+str(x)+' '+str(y)+' '+str(z)+' '+str(r)+'\n'
            # print('line: {}'.format(label_line))

            label_file=os.path.join(LABEL_FOLDER,name+'.txt')
            with open(label_file,'a+') as lf:
                lf.write(label_line)
            test_all_file=os.path.join(DATASET_FOLDER,'test.txt')
            with open(test_all_file,'a+') as tf:
                tf.write(name+'\n')