import cv2
import os
import time
file_path = 'detect_pics/00001.png'
os.remove(file_path)
# wait_num = 0
# while True:
#     if os.path.exists(file_path):
#         img = cv2.imread(file_path)
#         cv2.imshow('test',img)
#         cv2.waitKey(0)
#         os.remove(file_path)
#     else:
#         time.sleep(0.01)
#         wait_num += 0.1
#     if wait_num>100:
#         cv2.destroyAllWindows()
#         break