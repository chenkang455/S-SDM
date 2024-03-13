# synthesize the rendered img sequence into the video to check whether the output is right
from utils import *
import os
folder = r'E:\Data\Nerf\blender_data'
import cv2

for object in os.listdir(folder):
    for sub_name in os.listdir(os.path.join(folder,object)):
        if os.path.isdir(os.path.join(folder,object,sub_name)) == False:
            continue
        print(os.path.join(folder,object,sub_name))
        file_list = os.listdir(os.path.join(folder,object,sub_name))
        img_list = []
        for file in file_list:
            file_name = os.path.join(folder,object,sub_name,file)
            img_list.append(cv2.imread(file_name))
        save_video(img_list,path = os.path.join(folder,object,sub_name),duration = 10)

