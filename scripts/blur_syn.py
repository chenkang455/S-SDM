import os
import cv2
import numpy as np
from tqdm import trange
import argparse
from torchvision import transforms

#? convert the imgs under sharp_folder to blurry imgs on blur_folder
#? Structure as:
#?  base_folder
#?  ├── blur_folder
#?  ├── sharp_folder
#?  └── spike_folder

# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sharp_folder', type=str, default='E:/chenk/Data/test/raw_data')
    parser.add_argument('--img_type', type=str, default='.png')
    parser.add_argument('--overlap_len', type=int, default= 7)
    parser.add_argument('--height', type=int, default= 720)
    parser.add_argument('--width', type=int, default= 1280)
    parser.add_argument('--use_resize', action='store_true')
    parser.add_argument('--blur_num', type=int, default= 13)
    

    opt = parser.parse_args()
    width = opt.width
    height = opt.height
    use_resize = opt.use_resize
    sharp_folder = opt.sharp_folder
    base_folder = os.path.dirname(sharp_folder)
    blur_folder = os.path.join(base_folder,'blur_data')
    os.makedirs(blur_folder,exist_ok = True)
    
    for dirpath, sub_dirs, sub_files in os.walk(sharp_folder):
        if len(sub_files) == 0 or sub_files[0].endswith(opt.img_type) == False:
            continue
        print(dirpath)
        output_folder = dirpath.replace(sharp_folder,blur_folder)
        os.makedirs(output_folder,exist_ok = True)
        sub_files = sorted(sub_files)
        bais = 0 # bais that overlap between two blurry imgs
        num_blur_raw = opt.blur_num # number of sharp imgs before interpolated per blurry one
        num_inter = 7 # number of interpolated imgs between two imgs
        num_blur = (num_blur_raw - 1) * (num_inter + 1) + 1 # number of interpolated imgs per blurry one
        str_len = len(sub_files[0].split('.')[0])
        imgs = []
        start = 0
        for i in trange(len(sub_files)):
            if i + bais >= len(sub_files):
                break
            file_name = sub_files[i + bais]
            img = cv2.imread(os.path.join(dirpath,file_name))
            imgs.append(img)
            # synthesize the blurry image
            if i % (num_blur) == num_blur - 1:
                blur_img = np.mean(np.stack(imgs,axis = 0),axis = 0)
                end = i + bais
                # (num_blur_raw - 1 + (opt.overlap_len + 1) // (num_inter + 1)): number of imgs per blur
                # ((i + 1) // num_blur - 1): denotes the order of blur, 0 at first
                # num_blur_raw // 2: middle frame
                if use_resize:
                    blur_img = cv2.resize(blur_img,(width // 2,height//2),interpolation=cv2.INTER_LINEAR)
                print(f"Synthesize blurry {str((num_blur_raw - 1 + (opt.overlap_len + 1) // (num_inter + 1)) * ((i + 1) // num_blur - 1) + num_blur_raw // 2).zfill(str_len)}.png from {sub_files[start]} to {sub_files[end]}")
                cv2.imwrite(os.path.join(output_folder,f"{str((num_blur_raw - 1 + (opt.overlap_len + 1) // (num_inter + 1)) * ((i + 1) // num_blur - 1) + num_blur_raw // 2).zfill(str_len)}.png"),blur_img)
                imgs = []
                bais += opt.overlap_len
                start = i + bais + 1
                
