# used to convert the png files with blank background to black background
import os
import cv2
import numpy as np
from tqdm import trange
import shutil
# folder of the imgs rendered by blender 
blender_folder = r'D:\chenk\synthetic_data\blender_data'
base_folder = os.path.dirname(blender_folder)
sharp_folder = os.path.join(base_folder,'sharp_data')
# set the background as white or black
white = False
os.makedirs(sharp_folder,exist_ok = True)
for object_name in os.listdir(blender_folder):
    # if object_name != 'lego':
        # continue
    print(f"converting {object_name}")
    object_folder = os.path.join(blender_folder,object_name)
    for sub_name in os.listdir(object_folder):
        sub_folder = os.path.join(object_folder,sub_name)
        if os.path.isdir(sub_folder) == False:
            continue
        # data to be interpolated by XVFI 
        back_folder = os.path.join(sharp_folder,object_name,sub_name)
        print(back_folder)
        file_list = os.listdir(sub_folder)
        os.makedirs(back_folder,exist_ok = True)
        for i in trange(len(file_list)):
            # convert the transparent image to the image with white or black background 
            file_name = file_list[i]
            img = cv2.imread(os.path.join(sub_folder,file_name),cv2.IMREAD_UNCHANGED) #! directly converting the png to jpg has a small bug 
            if white:
                img = img[:,:,:3] * (img[:,:,3:4] / 255) + 255  * (1 - img[:,:,3:4] / 255)
            else:
                img = img[:,:,:3] * (img[:,:,3:4] / 255) + 0  * (1 - img[:,:,3:4] / 255)
            cv2.imwrite(os.path.join(back_folder,str(i).zfill(4) + ".png"),img)

