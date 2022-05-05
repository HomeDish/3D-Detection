# import os
# import sys
# import shutil
# file = 'data5'
# import argparse
# parser = argparse.ArgumentParser(description='consist time and rename file')
# parser.add_argument('--file',default= file)
# args = parser.parse_args()
#
# lidar_name = args.file
#
# csv_path = os.path.join('../../record',lidar_name)
# txt_path = os.path.join('.',lidar_name)
# csv_target_path = r'../dataset/kitti/training/velodyne'
# txt_target_path = r'../dataset/kitti/training/label_2'
# #%%
# csv_files = os.listdir(csv_path)
# txt_files = os.listdir(txt_path)
# print('txt_files num: ',len(txt_files))
# print('csv_files num: ',len(csv_files))
# #%%
# consist_times = []
# for file in txt_files:
#    if (file[:-3]+'csv') in csv_files:
#        consist_times.append(file[:-3])
# consist_times.sort()
# print('consist_times: ',len(consist_times))
# #%%
# for i in range(len(consist_times)):
#     csv_src_path = os.path.join(csv_path,consist_times[i]+'csv')
#     txt_src_path = os.path.join(txt_path,consist_times[i]+'txt')
#     # os.system('cp {} {}'.format(csv_src_path,csv_target_path))
#     shutil.copy(csv_src_path,csv_target_path)
#     shutil.copy(txt_src_path,txt_target_path)
#
#     try:
#         os.rename(os.path.join(csv_target_path,consist_times[i]+'csv'),os.path.join(csv_target_path,'{:06d}.csv'.format(i)))
#     except:
#         os.remove(os.path.join(csv_target_path,'{:06d}.csv'.format(i)))
#         os.rename(os.path.join(csv_target_path,consist_times[i]+'csv'),os.path.join(csv_target_path,'{:06d}.csv'.format(i)))
#     try:
#         os.rename(os.path.join(txt_target_path,consist_times[i]+'txt'),os.path.join(txt_target_path,'{:06d}.txt'.format(i)))
#     except:
#         os.remove(os.path.join(txt_target_path,'{:06d}.txt'.format(i)))
#         os.rename(os.path.join(txt_target_path,consist_times[i]+'txt'),os.path.join(txt_target_path,'{:06d}.txt'.format(i)))
#
# print('{} consistency and rename finished !!!!'.format(lidar_name))

import os
import sys
import shutil
files = ['data4','data5']
# csv_files = []
# txt_files = []
csv_target_path = r'../dataset/kitti/training/velodyne'
txt_target_path = r'../dataset/kitti/training/label_2'
start = 0; end = 0
for file in files:
    # file = 'data4'
    import argparse
    parser = argparse.ArgumentParser(description='consist time and rename file')
    parser.add_argument('--file',default= file)
    args = parser.parse_args()

    lidar_name = args.file
    csv_path = os.path.join('../../record', lidar_name)
    txt_path = os.path.join('.',lidar_name)

#%%
    csv_files=os.listdir(csv_path)
    txt_files=os.listdir(txt_path)
    print('txt_files num: ',len(txt_files))
    print('csv_files num: ',len(csv_files))
#%%
    consist_times = []
    for file in txt_files:
       if (file[:-3]+'csv') in csv_files:
           consist_times.append(file[:-3])
    consist_times.sort()
    print('consist_times: ',len(consist_times))

    end = start + len(consist_times)
    #%% copy and rename
    for i,idx in enumerate(range(start,end)):
        csv_src_path = os.path.join(csv_path,consist_times[i]+'csv')
        txt_src_path = os.path.join(txt_path,consist_times[i]+'txt')
        shutil.copy(csv_src_path,csv_target_path)
        shutil.copy(txt_src_path,txt_target_path)
        try:
            os.rename(os.path.join(csv_target_path,consist_times[i]+'csv'),os.path.join(csv_target_path,'{:06d}.csv'.format(idx)))
        except WindowsError:
            os.remove(os.path.join(csv_target_path,'{:06d}.csv'.format(idx)))
            os.rename(os.path.join(csv_target_path,consist_times[i]+'csv'),os.path.join(csv_target_path,'{:06d}.csv'.format(idx)))
        try:
            os.rename(os.path.join(txt_target_path,consist_times[i]+'txt'),os.path.join(txt_target_path,'{:06d}.txt'.format(idx)))
        except WindowsError:
            os.remove(os.path.join(txt_target_path,'{:06d}.txt'.format(idx)))
            os.rename(os.path.join(txt_target_path,consist_times[i]+'txt'),os.path.join(txt_target_path,'{:06d}.txt'.format(idx)))
    start = end
print('{} consistency and rename finished, generate {} files !!!!'.format(files,end))
### amend tarin.txt and val.txt in ImageSets
LEN = end
imageset_path = '../dataset/kitti/ImageSets'
with open(os.path.join(imageset_path,'train.txt'),'w') as f_train:
    for i in range(LEN):
        f_train.write('{:06d}\n'.format(i))
with open(os.path.join(imageset_path,'val.txt'),'w') as f_val:
    for i in range(LEN):
        f_val.write('{:06d}\n'.format(i))
