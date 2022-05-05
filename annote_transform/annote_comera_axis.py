#coding=utf-8
import yaml
import os
#单个文档
import argparse
parser = argparse.ArgumentParser(description='Convert yaml to KITTI annotation')
file = 'data5'
parser.add_argument('--file',default= file)
args = parser.parse_args()
def get_yaml_data(yaml_file):
    #打开yaml文件
    print("***获取yaml文件数据***")
    file=open(yaml_file,'r',encoding='utf-8')
    file_data=file.read()
    file.close()

    # print(file_data)
    # print("类型",type(file_data))

    #将字符串转化为字典或列表
    # print("***转化yaml数据为字典或列表***")
    data=yaml.safe_load(file_data)  #safe_load，safe_load,unsafe_load
    # print(data)
    # print("类型",type(data))
    return data
map_label = {
    'person':'Pedestrian',
    'car':'Car',
    'bicycle':'Cyclist',
    '':'DontCare',
}
file = args.file
if not os.path.exists('{}'.format(file)):
    os.mkdir('{}'.format(file))
#####
current_path=os.path.abspath(".")
yaml_path=os.path.join(current_path,file+".yaml")
data = get_yaml_data(yaml_path)
#####
# data['tracks'][0]['track'][0]
use_data = data['tracks']
tracks_N = len(use_data)
for i in range(tracks_N):
    track_data = use_data[i]
    temp_i = track_data['track']
    an_N = len(temp_i)
    for j in range(an_N):
        temp_j = temp_i[j]
        time_sec = temp_j['header']['stamp']
        an_file_name = str(time_sec['secs'])+'.'+str(time_sec['nsecs'])
        #### write_info
        info_0 = map_label[temp_j['label']]
        info_1 = '0.00'
        info_2 = '0'
        info_3 = '0'
        info_4 = '0.0'
        info_5 = '0.0'
        info_6 = '0.0'
        info_7 = '0.0'
        ### info_8,info_9,info_10 have been validated efficient——liujialai,2021-11-26
        info_8 = str(temp_j['box']['height'])
        info_9 = str(temp_j['box']['width'])
        info_10 = str(temp_j['box']['length'])
        info_11 = str(temp_j['translation']['y']*(-1))
        info_12 = str(temp_j['translation']['z']*(-1))
        info_13 = str(temp_j['translation']['x'])
        info_14 = '1.00'
        with open('{}/{}.txt'.format(file,an_file_name),'a') as f:
            write_data = ' '.join([info_0,info_1,info_2,info_3,info_4,
                                   info_5,info_6,info_7,info_8,
                                   info_9,info_10,info_11,info_12,
                                   info_13,info_14])
            f.write('{}\n'.format(write_data))
print('{} finished！'.format(file))
