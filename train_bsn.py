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
from torch.optim.lr_scheduler import CosineAnnealingLR,MultiStepLR
from codes.utils import * 
from codes.dataset import *
from codes.metrics import compute_img_metric
from codes.model.bsn_model import BSN


# SDM model
def sdm_deblur(blur,spike,spike_idx):
    """ Basic SDM model.

    Args:
        blur (tensor): blurry input. [bs,3,w,h]
        spike (tensor): spike sequence. [bs,137,w,h]
        spike_idx (int): central idx of the short-exposure spike stream

    Returns:
        rgb_sdm: deblur result. [bs,3,w,h]
    """
    global spike_bsn_len
    spike_sum = torch.sum(spike[:,20:-20],dim = 1,keepdim = True)
    spike_bsn = spike[:,spike_idx - spike_bsn_len // 2:spike_idx + spike_bsn_len // 2 + 1,:,:]
    rgb_sdm = blur / spike_sum
    rgb_sdm[spike_sum.repeat(1,3,1,1) == 0] = 0
    rgb_sdm = rgb_sdm * torch.sum(spike_bsn,dim = 1,keepdim = True) * 97 / spike_bsn_len 
    rgb_sdm = rgb_sdm.clip(0,1)
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

if __name__ == '__main__':
    # parameters 
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,default=r'GOPRO/',help = 'base folder of the GOPRO dataset')
    parser.add_argument('--save_folder', type=str,default='exp/BSN', help = 'experimental results save folder')
    parser.add_argument('--data_type',type=str, default='GOPRO' ,help = 'dataset type')
    parser.add_argument('--exp_name', type=str,default='test', help = 'experiment name')
    parser.add_argument('--epochs', type=int, default=1001)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bsn_len', type=int, default=9, help = 'spike length for BSN input TFP image')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--use_small', action='store_true',default = False,help='train at the small GOPRO dataset for debugging')
    parser.add_argument('--spike_full', action='store_true',default = False,help='train BSN under high resolution spike stream (1280 * 720)')
    parser.add_argument('--test_mode', action='store_true',default = False, help='test the metric')
    parser.add_argument('--bsn_path', type=str,default='model/BSN_1000.pth')
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
    resize_method = transforms.Resize((opt.height // 4,opt.width // 4),interpolation=transforms.InterpolationMode.BILINEAR)
    logger.info(opt)
    
    # train and test data splitting
    train_dataset = SpikeData(opt.base_folder,opt.data_type,'train',use_roi = True,
                              roi_size = [128 * 4,128 * 4],use_small = opt.use_small,spike_full = opt.spike_full)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=4,pin_memory=True)
    test_dataset = SpikeData(opt.base_folder,opt.data_type,'test',use_roi = False,
                             use_small = opt.use_small,spike_full = opt.spike_full)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=1,pin_memory=True)
    
    # config for network and training parameters
    bsn_teacher = BSN(n_channels = 1,n_output = 1).cuda()
    if opt.test_mode:
        bsn_teacher.load_state_dict(torch.load(opt.bsn_path))
    optim = Adam(bsn_teacher.parameters(), lr=opt.lr)
    scheduler = CosineAnnealingLR(optim, T_max=opt.epochs) 

    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spike_bsn_len = opt.bsn_len
    # -------------------- train ----------------------  
    train_start = datetime.now()
    logger.info("Start Training!")
    for epoch in range(opt.epochs):
        if opt.test_mode == False:
            train_loss = AverageMeter()
            for batch_idx, (blur,spike,sharp) in enumerate(tqdm(train_loader)):
                # read the data
                blur,spike = blur.to(device),spike.to(device)
                # reconstruct the initial result
                for spike_idx in range(20,117,3 * 8):
                    # TFP Part
                    tfp = cal_tfp(spike,spike_idx,spike_bsn_len)
                    tfp_bsn = bsn_teacher(tfp).clip(0,1)
                    loss = criterion(tfp,tfp_bsn)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    writer.add_scalar('Training Loss', loss.item())
                    train_loss.update(loss.item())
            logger.info(f"EPOCH {epoch}/{opt.epochs}: Train Loss: {train_loss.avg}")
            writer.add_scalar('Epoch Loss', train_loss.avg, epoch)
            scheduler.step()
        # visualization result
        if epoch % 100 == 0:
            with torch.no_grad():
                # save the network
                save_network(bsn_teacher, f"{ckpt_folder}/BSN_{epoch}.pth")
                # visualization
                for batch_idx, (blur,spike,sharp) in enumerate(tqdm(test_loader)):
                    if batch_idx in [int(i) for i in np.linspace(0,len(test_loader),5)]:
                        blur,spike = blur.to(device),spike.to(device)
                        spike_idx = len(spike[0]) // 2 
                        # TFP Part
                        tfp = cal_tfp(spike,spike_idx,spike_bsn_len)
                        tfp_bsn = bsn_teacher(tfp).clip(0,1)
                        # visualization
                        save_img(img = normal_img(blur[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_blur.png')
                        save_img(img = normal_img(sharp[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_sharp.png')
                        save_img(img = normal_img(tfp[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tfp.png')
                        save_img(img = normal_img(tfp_bsn[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tfp_bsn.png')
                    else:
                        continue
        # save metric result
        if epoch % 100 == 0:
            with torch.no_grad():
                # calculate the metric
                metrics = {}
                method_list = ['TFP','BSN']
                metric_list = ['mse','ssim','psnr','lpips']
                for method_name in method_list:
                    metrics[method_name] = {}  # 初始化每个方法的字典
                    for metric_name in metric_list:
                        metrics[method_name][metric_name] = AverageMeter()
                for batch_idx, (blur,spike,sharp) in enumerate(tqdm(test_loader)):
                    blur,spike = blur.to(device),spike.to(device)
                    sharp = 0.11 * sharp[:,0:1] + 0.59 * sharp[:,1:2] + 0.3 * sharp[:,2:3]
                    sharp = resize_method(sharp)
                    spike_idx = len(spike[0]) // 2 
                    # TFP and BSN
                    tfp = cal_tfp(spike,spike_idx,spike_bsn_len)
                    tfp_bsn = bsn_teacher(tfp).clip(0,1)
                    # Metric
                    for key in metric_list :
                        metrics['TFP'][key].update(compute_img_metric(tfp,sharp,key))
                        metrics['BSN'][key].update(compute_img_metric(tfp_bsn,sharp,key))
                # Print all results
                for method_name in method_list:
                    re_msg = ''
                    for metric_name in metric_list:
                        re_msg += metric_name + ": " + "{:.3f}".format(metrics[method_name][metric_name].avg) + "  "
                    logger.info(f"{method_name}: " + re_msg)
                    writer.add_scalar(f'{method_name}/{metric_name}', metrics[method_name][metric_name].avg, epoch)
        # test mode
        if opt.test_mode:
                break
    writer.close()

