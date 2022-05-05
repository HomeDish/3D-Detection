import os
import shutil
csv_origin_path = os.path.join('../../','record','data5_rostime')
csv_target_path = os.path.join('../dataset/kitti/testing/velodyne')
## clear csv_target_path
for file in os.listdir(csv_target_path):
    os.remove(os.path.join(csv_target_path,file))
## copy origin csv to target
for file in os.listdir(csv_origin_path):
    shutil.copy(os.path.join(csv_origin_path,file),os.path.join(csv_target_path,file))
## rename
for idx,file in enumerate(sorted(os.listdir(csv_target_path))):
    os.rename(os.path.join(csv_target_path,file),os.path.join(csv_target_path, '{:06d}.csv'.format(idx)))
## amend test.txt in ImageSets
LEN = len(os.listdir(csv_origin_path))
imageset_path = '../dataset/kitti/ImageSets'
with open(os.path.join(imageset_path,'test.txt'),'w') as f_test:
    for i in range(LEN):
        f_test.write('{:06d}\n'.format(i))
print('there are {} testing samples'.format(LEN))