import os
import sys
path_csv=r"/home/bdsc/Desktop/Lidar/record/data4_csv/"

path_txt=r'/home/bdsc/Desktop/Lidar/record/data4_txt/'


#获取该目录下所有文件，存入列表中
file_txt_List=os.listdir(path_txt)
file_csv_List=os.listdir(path_csv)
#sys.path.append(path_csv1)
#delete csv
# file_de=list(set(file_csv_List)-set(file_txt_List))
# for file_name in file_de:
#     de_file = path_csv+file_name
#     os.remove(de_file)
#delete txt
file_de=list(set(file_txt_List)-set(file_csv_List))
for file_name in file_de:
    de_file = path_txt+file_name
    os.remove(de_file)