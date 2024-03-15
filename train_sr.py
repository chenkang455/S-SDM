import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import os
import argparse
from datetime import datetime
from torch.optim import Adam
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil

from codes.dataset import *
from codes.utils import * 
from codes.metrics import compute_img_metric
from codes.model.bsn_model import BSN
from codes.model.edsr_model import EDSR


# SDM model
def sdm_deblur(blur,tfp_long,tfp_short):
    """ General SDM model.

    Args:
        blur (tensor): blurry input. [bs,3,w,h]
        tfp_long (tensor): long tfp corresponding to the blurry input. [bs,1,w,h]
        tfp_short (tensor): short tfp corresponding to the short-exposure image. [bs,1,w,h]

    Returns:
        rgb_sdm: deblur result. [bs,3,w,h]
    """
    tfp_long = tfp_long.repeat(1,3,1,1)
    tfp_short = tfp_short.repeat(1,3,1,1)
    rgb_sdm = blur / tfp_long
    rgb_sdm[tfp_long == 0] = 0
    rgb_sdm = rgb_sdm * tfp_short
    return rgb_sdm

# TFP model
def cal_tfp(spike,spike_idx,tfp_len):
    """TPF Model

    Args:
        spike (tensor): spike sequence. [bs,137,w,h]
        spike_idx (int): central idx of the virtual exposure window
        tfp_len (_type_): length of the virtual exposure window. 97 for long-TFP, [7,9,11] for short-TFP.

    Returns:
        tfp_pred: tfp result
    """
    spike = spike[:,spike_idx - tfp_len // 2:spike_idx + tfp_len // 2 + 1,:,:]
    tfp_pred = torch.mean(spike,dim = 1,keepdim = True)
    return tfp_pred

# main function
if __name__ == '__main__':
    # parameters 
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,default=r'GOPRO')
    parser.add_argument('--save_folder', type=str,default='exp/SR')
    parser.add_argument('--data_type',type=str, default='GOPRO')
    parser.add_argument('--exp_name', type=str,default='test')
    parser.add_argument('--bsn_path', type=str,default='model/BSN_1000.pth')
    parser.add_argument('--sr_path', type=str,default='model/SR_70.pth')
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bsn_len', type=int, default=9)
    parser.add_argument('--use_small', action='store_true',default = False)
    parser.add_argument('--test_mode', action='store_true',default = False)
    opt = parser.parse_args()
    
    # prepare
    ckpt_folder = f"{opt.save_folder}/{opt.exp_name}/ckpts"
    img_folder = f"{opt.save_folder}/{opt.exp_name}/imgs"
    os.makedirs(ckpt_folder,exist_ok= True)
    os.makedirs(img_folder,exist_ok= True)
    set_random_seed(opt.seed)
    save_opt(opt,f"{opt.save_folder}/{opt.exp_name}/opt.txt")
    log_file = f"{opt.save_folder}/{opt.exp_name}/results.txt"
    logger = setup_logging(log_file)
    if os.path.exists(f'{opt.save_folder}/{opt.exp_name}/tensorboard'):
        shutil.rmtree(f'{opt.save_folder}/{opt.exp_name}/tensorboard')
    writer = SummaryWriter(f'{opt.save_folder}/{opt.exp_name}/tensorboard')
    logger.info(opt)
    
    # train and test data splitting
    train_dataset = SpikeData(opt.base_folder,opt.data_type,'train',use_roi = True,
                              roi_size = [128 * 4,128 * 4],use_small = opt.use_small)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=4,pin_memory=True)
    test_dataset = SpikeData(opt.base_folder,opt.data_type,'test',use_roi = False,use_small = opt.use_small)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=1,pin_memory=True)
    
    # config for network and training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr_net = EDSR(color_num = 1).to(device)
    if opt.test_mode:
        sr_net.load_state_dict(torch.load(opt.sr_path))
    optim = Adam(sr_net.parameters(), lr=opt.lr)
    bsn_net = BSN(n_channels=1, n_output=1).to(device)
    bsn_net.load_state_dict(torch.load(opt.bsn_path))
    for param in bsn_net.parameters():
        param.requires_grad = False
    criterion = nn.MSELoss()
    spike_bsn_len = opt.bsn_len
    resize_method = transforms.Resize((720,1280),interpolation=transforms.InterpolationMode.NEAREST)
    # -------------------- train ----------------------  
    train_start = datetime.now()
    logger.info("Start Training!")
    for epoch in range(opt.epochs):
        train_loss = AverageMeter()
        if opt.test_mode == False:
            for batch_idx, (blur,spike,sharp) in enumerate(tqdm(train_loader)):
                # read the data
                blur = 0.11 * blur[:,0:1] + 0.59 * blur[:,1:2] + 0.3 * blur[:,2:3]
                blur,spike = blur.to(device),spike.to(device)
                for spike_idx in [len(spike[0]) // 2]:
                    # Long-TFP Part
                    tfp_long = cal_tfp(spike,spike_idx,97)
                    sr_tfp = sr_net(tfp_long).clip(0,1)
                    loss = criterion(sr_tfp,blur)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    writer.add_scalar('Training Loss', loss.item())
                    train_loss.update(loss.item())
        logger.info(f"EPOCH {epoch}/{opt.epochs}: Train Loss: {train_loss.avg}")
        writer.add_scalar('Epoch Loss', train_loss.avg, epoch)
        # visualization result
        if epoch % 5 == 0:
            with torch.no_grad():
                # save the network
                save_network(sr_net, f"{ckpt_folder}/SR_{epoch}.pth")
                # visualization
                for batch_idx, (blur,spike,sharp) in enumerate(tqdm(test_loader)):
                    if batch_idx in [i for i in range(0,len(test_loader),200)]:
                        blur,spike = blur.to(device),spike.to(device)
                        spike_idx = len(spike[0]) // 2 
                        # Long-TFP Part
                        tfp_long = cal_tfp(spike,spike_idx,97)
                        tfp_long_sr = sr_net(tfp_long).clip(0,1)
                        save_img(img = normal_img(blur[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_blur.png')
                        save_img(img = normal_img(sharp[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_sharp.png')
                        save_img(img = normal_img(tfp_long[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tfp_long.png')
                        save_img(img = normal_img(tfp_long_sr[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tfp_long_sr.png')
                        # Short-TFP Part
                        tfp = cal_tfp(spike,spike_idx,spike_bsn_len)
                        tfp_resize = resize_method(tfp).clip(0,1)
                        tfp_bsn = bsn_net(tfp).clip(0,1)
                        tfp_bsn_resize = resize_method(tfp_bsn).clip(0,1)
                        tfp_bsn_sr = sr_net(tfp_bsn).clip(0,1)
                        deblur = sdm_deblur(blur,tfp_long_sr,tfp_bsn_sr)
                        save_img(img = normal_img(tfp[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tfp_short.png')
                        save_img(img = normal_img(tfp_bsn[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tfp_short_bsn.png')
                        save_img(img = normal_img(tfp_bsn_sr[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tfp_short_bsn_sr.png')
                        save_img(img = normal_img(tfp_bsn_resize[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tfp_short_bsn_resize.png')
                        save_img(img = normal_img(tfp_resize[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tfp_short_resize.png')
                        save_img(img = normal_img(deblur[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_deblur.png')
                        if opt.test_mode:
                            break
                    else:
                        continue
        # save metric result
        if epoch % 10 == 0:
            with torch.no_grad():
                # calculate the metric
                metrics = {}
                method_list = ['SDM','Blur_SR','Blur_Resize','BSN_SR','BSN_Resize','TFP_Resize']
                # metric_list = ['mse','ssim','psnr','lpips']
                metric_list = ['ssim','psnr']
                for method_name in method_list:
                    metrics[method_name] = {}  # 初始化每个方法的字典
                    for metric_name in metric_list:
                        metrics[method_name][metric_name] = AverageMeter()
                for batch_idx, (blur,spike,sharp) in enumerate(tqdm(test_loader)):
                    blur,spike = blur.to(device),spike.to(device)
                    blur_gray = 0.11 * blur[:,0:1] + 0.59 * blur[:,1:2] + 0.3 * blur[:,2:3]
                    sharp_gray = 0.11 * sharp[:,0:1] + 0.59 * sharp[:,1:2] + 0.3 * sharp[:,2:3]
                    spike_idx = len(spike[0]) // 2 
                    # Metric
                    # Long-TFP Part
                    tfp_long = cal_tfp(spike,spike_idx,97)
                    tfp_long_resize = resize_method(tfp_long).clip(0,1)
                    tfp_long_sr = sr_net(tfp_long).clip(0,1)
                    # Short-TFP Part
                    tfp = cal_tfp(spike,spike_idx,spike_bsn_len)
                    tfp_resize = resize_method(tfp).clip(0,1)
                    tfp_bsn = bsn_net(tfp).clip(0,1)
                    tfp_bsn_resize = resize_method(tfp_bsn).clip(0,1)
                    tfp_bsn_sr = sr_net(tfp_bsn).clip(0,1)
                    deblur = sdm_deblur(blur,tfp_long_sr,tfp_bsn_sr)
                    for key in metric_list :
                        # SDM
                        metrics['SDM'][key].update(compute_img_metric(deblur,sharp,key))
                        # BLUR
                        metrics['Blur_SR'][key].update(compute_img_metric(tfp_long_sr,blur_gray,key))
                        metrics['Blur_Resize'][key].update(compute_img_metric(tfp_long_resize,blur_gray,key))
                        # BSN
                        metrics['BSN_SR'][key].update(compute_img_metric(tfp_bsn_sr,sharp_gray,key))
                        metrics['BSN_Resize'][key].update(compute_img_metric(tfp_bsn_resize,sharp_gray,key))
                        # TFP
                        metrics['TFP_Resize'][key].update(compute_img_metric(tfp_resize,sharp_gray,key))

                # Print all results
                for method_name in method_list:
                    re_msg = ''
                    for metric_name in metric_list:
                        re_msg += metric_name + ": " + "{:.3f}".format(metrics[method_name][metric_name].avg) + "  "
                    logger.info(f"{method_name}: " + re_msg)
                    writer.add_scalar(f'{method_name}/{metric_name}', metrics[method_name][metric_name].avg, epoch)
        # stop
        if opt.test_mode:
            break
    writer.close()