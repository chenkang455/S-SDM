import os
import cv2
import numpy as np
from tqdm import trange
import shutil
from utils import *
import shutil
# folder params
blender_folder = r'D:\chenk\synthetic_raw_data\blender_data'
base_folder = os.path.dirname(blender_folder)
final_folder = os.path.join(base_folder,'syn_data')
blur_name = 'blur_data'
sharp_name = 'sharp_data'
spike_name = 'spike_data'
blender_name = 'blender_data'
white = False

for object_name in os.listdir(blender_folder):
    # if object_name != 'drums':
    #     continue
    print(object_name)
    make_folder(os.path.join(final_folder,object_name))
    for sub_name in os.listdir(os.path.join(blender_folder,object_name)):
        if os.path.isdir(os.path.join(blender_folder,object_name,sub_name)) == False:
            shutil.copy(os.path.join(blender_folder,object_name,sub_name),os.path.join(final_folder,object_name,sub_name))
        else:
            make_folder(os.path.join(final_folder,object_name,sub_name))
            if os.path.exists(os.path.join(final_folder,object_name,sub_name,blur_name)):
                shutil.rmtree(os.path.join(final_folder,object_name,sub_name,blur_name))
            if os.path.exists(os.path.join(final_folder,object_name,sub_name,spike_name)):
                shutil.rmtree(os.path.join(final_folder,object_name,sub_name,spike_name))
            if os.path.exists(os.path.join(final_folder,object_name,sub_name,sharp_name)):
                shutil.rmtree(os.path.join(final_folder,object_name,sub_name,sharp_name))
            shutil.copytree(os.path.join(base_folder,blur_name,object_name,sub_name), os.path.join(final_folder,object_name,sub_name,blur_name))
            shutil.copytree(os.path.join(base_folder,spike_name,object_name,sub_name), os.path.join(final_folder,object_name,sub_name,spike_name))
            make_folder(os.path.join(final_folder,object_name,sub_name,sharp_name))
            file_list = os.listdir(os.path.join(base_folder,blur_name,object_name,sub_name))
            for i in range(len(file_list)):
                img = cv2.imread(os.path.join(base_folder,sharp_name,object_name,sub_name,file_list[i]),cv2.IMREAD_UNCHANGED) #! directly converting the png to jpg has a small bug 
                cv2.imwrite(os.path.join(final_folder,object_name,sub_name,sharp_name,file_list[i]),img)