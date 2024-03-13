import os
import cv2
import numpy as np
from tqdm import trange
import argparse

#? extract the imgs under raw_folder to sharp sequence on sharp_folder
#? Structure as:
#?  base_folder
#?  ├── blur_folder
#?  ├── raw_folder
#?  └── spike_folder

# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_folder', type=str, default=r'E:\chenk\Data\GOPRO\test\raw_data')
    parser.add_argument('--img_type', type=str, default='.png')
    parser.add_argument('--overlap_len', type=int, default= 7)
    parser.add_argument('--height', type=int, default= 720)
    parser.add_argument('--width', type=int, default= 1280)
    parser.add_argument('--use_resize', action='store_true')
    parser.add_argument('--blur_num', type=int, default= 13)
    parser.add_argument('--multi', action='store_true',default = False)
    
    opt = parser.parse_args()
    width = opt.width
    height = opt.height
    use_resize = opt.use_resize
    raw_folder = opt.raw_folder
    base_folder = os.path.dirname(raw_folder)
    blur_folder = os.path.join(base_folder,'blur_data')
    sharp_folder = os.path.join(base_folder,'sharp_data')
    
    for dirpath, sub_dirs, sub_files in os.walk(raw_folder):
        if len(sub_files) == 0 or sub_files[0].endswith(opt.img_type) == False:
            continue
        print(dirpath)
        output_folder = dirpath.replace(raw_folder,sharp_folder)
        os.makedirs(output_folder,exist_ok = True)
        sub_files = sorted(sub_files)
        bais = 0 # bais that overlap between two blurry imgs
        num_blur_raw = opt.blur_num # number of sharp imgs before interpolated per blurry one
        num_inter = 7 # number of interpolated imgs between two imgs
        num_blur = num_blur_raw * (num_inter + 1) + 1 # number of interpolated imgs per blurry one
        str_len = len(sub_files[0].split('.')[0])
        imgs = []
        idx = 0
        loop_bais = 0
        for i in trange(len(sub_files)):
            if i % (num_inter + 1) == 0:
                if idx % num_blur_raw == num_blur_raw // 2 or (opt.multi == True and ((idx - loop_bais) % 12) % 2 == 0):
                    file_name = sub_files[i]
                    print(file_name)
                    img = cv2.imread(os.path.join(dirpath,file_name))
                    if use_resize:
                        img = cv2.resize(img,(width // 2,height//2),interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(output_folder,f"{str(idx).zfill(str_len)}.png"),img)
                if idx % num_blur_raw == num_blur_raw - 1:
                    loop_bais += 1
                idx += 1
