import os
import sys
path0=r"/home/bdsc/Desktop/Lidar/record/data4_csv"
path1=r"/home/bdsc/Desktop/Lidar/record/data4_csv/"
sys.path.append(path1)
# print(sys.path)

# 列出当前目录下所有的文件
files = os.listdir(path0)

# files = os.listdir('.')

print('files',files)

for filename in files:
    portion = os.path.splitext(filename)
# 如果后缀是.txt
    if portion[1] == ".txt":
# 重新组合文件名和后缀名

        newname = portion[0] + ".csv"
        filenamedir=path1 +filename
        newnamedir=path1+newname

# os.rename(filename,newname)
        os.rename(filenamedir,newnamedir)