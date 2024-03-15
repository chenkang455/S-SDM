import cv2
import torch
import numpy as np
import imageio
import os
import torch.nn as nn
import random
# Save Network 
def save_network(network, save_path):
    if isinstance(network, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

def set_random_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)

def save_opt(opt,opt_path):
    with open(opt_path, 'w') as f:
        for key, value in vars(opt).items():
            f.write(f"{key}: {value}\n")

def save_gif(image_list, gif_path = 'test', duration = 2,RGB = True,nor = False):
    imgs = []
    os.makedirs('Video',exist_ok = True)
    with imageio.get_writer(os.path.join('Video',gif_path + '.gif'), mode='I',duration = 1000 * duration / len(image_list),loop=0) as writer:
        for i in range(len(image_list)):
            img = normal_img(image_list[i],RGB,nor)
            writer.append_data(img)

def save_video(image_list,path = 'test',duration = 2,RGB = True,nor = False):
    os.makedirs('Video',exist_ok = True)
    imgs = []
    for i in range(len(image_list)):
        img = normal_img(image_list[i],RGB,nor)
        imgs.append(img)
    imageio.mimwrite(os.path.join('Video',path + '.mp4'), imgs, fps=30, quality=8)


def normal_img(img,RGB = True,nor = True):
    if nor:
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
    if (img.shape[0] == 3 or img.shape[0] == 1) and isinstance(img,torch.Tensor):
        img = img.permute(1,2,0)
    if isinstance(img,torch.Tensor):
        img = np.array(img.detach().cpu())
    if len(img.shape) == 2:
        img = img[...,None]
    if img.shape[-1] == 1:
        img = np.repeat(img,3,axis = -1)
    img = img.astype(np.uint8)
    if RGB == False:
        img = img[...,::-1]
    return img

def save_img(path = 'test.png',img = None,nor = True):
    if nor:
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
    if isinstance(img,torch.Tensor):
        img = np.array(img.detach().cpu())
    img = img.astype(np.uint8)
    cv2.imwrite(path,img)

def make_folder(path):
    os.makedirs(path,exist_ok = True)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count

def video_to_spike(
    sourefolder=None,
    imgs = None, 
    savefolder_debug=None, 
    threshold=5.0,
    init_noise=True,
    format="png",
    ):
    """
        函数说明
        :param 参数名: 参数说明
        :return: 返回值名: 返回值说明
    """
    if sourefolder != None:
        filelist = sorted(os.listdir(sourefolder))
        datas = [fn for fn in filelist if fn.endswith(format)]
        
        T = len(datas)
        
        frame0 = cv2.imread(os.path.join(sourefolder, datas[0]))
        H, W, C = frame0.shape

        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

        spikematrix = np.zeros([T, H, W], np.uint8)

        if init_noise:
            integral = np.random.random(size=([H,W])) * threshold
        else:
            integral = np.random.zeros(size=([H,W]))
        
        Thr = np.ones_like(integral).astype(np.float32) * threshold

        for t in range(0, T):
            frame = cv2.imread(os.path.join(sourefolder, datas[t]))
            if C > 1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray / 255.0
            integral += gray
            fire = (integral - Thr) >= 0
            fire_pos = fire.nonzero()
            
            integral[fire_pos] -= threshold
            spikematrix[t][fire_pos] = 1

        if savefolder_debug:
            np.save(os.path.join(savefolder_debug, "spike_debug.npy"), spikematrix)
    elif imgs != None:
        frame0 = imgs[0]
        H, W, C = frame0.shape
        T = len(imgs)
        spikematrix = np.zeros([T, H, W], np.uint8)

        if init_noise:
            integral = np.random.random(size=([H,W])) * threshold
        else:
            integral = np.random.zeros(size=([H,W]))
        
        Thr = np.ones_like(integral).astype(np.float32) * threshold

        for t in range(0, T):
            frame = imgs[t]
            if C > 1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray / 255.0
            integral += gray
            fire = (integral - Thr) >= 0
            fire_pos = fire.nonzero()
            
            integral[fire_pos] -= threshold
            spikematrix[t][fire_pos] = 1
    return spikematrix


def load_vidar_dat(filename, left_up=(0, 0), window=None, frame_cnt = None,height = 800, width = 800, **kwargs):
    if isinstance(filename, str):
        array = np.fromfile(filename, dtype=np.uint8)
    elif isinstance(filename, (list, tuple)):
        l = []
        for name in filename:
            a = np.fromfile(name, dtype=np.uint8)
            l.append(a)
        array = np.concatenate(l)
    else:
        raise NotImplementedError

    if window == None:
        window = (height - left_up[0], width - left_up[0])

    len_per_frame = height * width // 8
    framecnt = frame_cnt if frame_cnt != None else len(array) // len_per_frame

    spikes = []

    for i in range(framecnt):
        compr_frame = array[i * len_per_frame: (i + 1) * len_per_frame]
        blist = []
        for b in range(8):
            blist.append(np.right_shift(np.bitwise_and(compr_frame, np.left_shift(1, b)), b))
        
        frame_ = np.stack(blist).transpose()
        frame_ = np.flipud(frame_.reshape((height, width), order='C'))

        if window is not None:
            spk = frame_[left_up[0]:left_up[0] + window[0], left_up[1]:left_up[1] + window[1]]
        else:
            spk = frame_

        spk = spk.copy().astype(np.float32)[None]

        spikes.append(spk)

    return np.concatenate(spikes)


import logging
# log info
def setup_logging(log_file):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='w')  # 使用'w'模式打开文件
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
    
    
def generate_labels(file_name):
    num_part = file_name.split('/')[-1]
    non_num_part = file_name.replace(num_part, '')
    num = int(num_part)
    labels = [non_num_part + str(num + 2 * i).zfill(len(num_part)) + '.png' for i in range(-3, 4)]
    return labels