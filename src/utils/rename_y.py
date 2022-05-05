import os

path = '/home/bdsc/Desktop/Lidar/record/data_all_txt/data4_txt'

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
print(fileList)
n = 0
for i in fileList:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符
    width = 6  # 指定宽度
    # 设置新文件名
    newname = path + os.sep + (str(n).zfill(width)) + '.txt'

    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

    n += 1