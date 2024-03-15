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
from codes.utils import * 
from codes.metrics import compute_img_metric
import glob
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from codes.dataset import *
from codes.model.ldn_model import Deblur_Net
from codes.model.bsn_model import BSN
from codes.model.edsr_model import EDSR
import lpips
from pytorch_msssim import SSIM
from torch.optim.lr_scheduler import CosineAnnealingLR

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

if __name__ == '__main__':
    # parameters 
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', default='GOPRO/')
    parser.add_argument('--save_folder', default='exp/Deblur')
    parser.add_argument('--data_type', default='GOPRO')
    parser.add_argument('--exp_name', default='test')
    parser.add_argument('--bsn_path', default='model/BSN_1000.pth')
    parser.add_argument('--sr_path', default='model/SR_70.pth')
    parser.add_argument('--deblur_path', default='model/DeblurNet_100.pth')
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spike_deblur_len', type=int, default=21)
    parser.add_argument('--spike_bsn_len', type=int, default=9)
    parser.add_argument('--lambda_tea', type=float, default=1)
    parser.add_argument('--lambda_reblur', type=float, default=100)
    parser.add_argument('--blur_step', type=int, default=24)
    parser.add_argument('--use_small', action='store_true',default = False)
    parser.add_argument('--test_mode', action='store_true',default = False)
    parser.add_argument('--use_ssim', action='store_true',default = False, help= 'use ssim loss or not')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--roi_size', type=int, default= 512)
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
    train_dataset = SpikeData(opt.base_folder,opt.data_type,'test',use_roi = True, roi_size = [opt.roi_size,opt.roi_size],use_small= opt.use_small)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=4,pin_memory=True)
    test_dataset = SpikeData(opt.base_folder,opt.data_type,'test',use_roi = False,use_small= opt.use_small)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=1,pin_memory=True)
    
    # config for network and training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # BSN
    bsn_net = BSN(n_channels = 1,n_output = 1).to(device)
    bsn_net.load_state_dict(torch.load(opt.bsn_path))
    for param in bsn_net.parameters():
        param.requires_grad = False
    # SR
    sr_net = EDSR(color_num = 1).to(device)
    sr_net.load_state_dict(torch.load(opt.sr_path))
    for param in sr_net.parameters():
        param.requires_grad = False
    # Deblur
    spike_bsn_len = opt.spike_bsn_len
    spike_deblur_len = opt.spike_deblur_len
    deblur_net = Deblur_Net(spike_dim = spike_deblur_len).to(device)
    if opt.test_mode == True:
        deblur_net.load_state_dict(torch.load(opt.deblur_path))
    # other settting
    optim = Adam(deblur_net.parameters(), lr=opt.lr)
    scheduler = CosineAnnealingLR(optim, T_max=opt.epochs) 
    loss_lpips = lpips.LPIPS(net='vgg').to(device)
    loss_ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    loss_mse = nn.MSELoss()
    # -------------------- train ----------------------  
    train_start = datetime.now()
    logger.info("Start Training!")
    for epoch in range(opt.epochs):
        # loss definition
        train_loss_all = AverageMeter()
        tea_loss_all = AverageMeter()
        reblur_loss_all = AverageMeter()
        if opt.test_mode == False:
            for batch_idx, (blur,spike,sharp) in enumerate(tqdm(train_loader)):
                # read the data
                blur,spike = blur.to(device),spike.to(device)
                # reconstruct the initial result
                train_loss = 0
                tea_loss = 0
                reblur_loss = 0
                reblur = []
                spike_start,start_end,spike_step = 20,117,opt.blur_step
                spike_num = (start_end - spike_start - 1) / spike_step + 1
                for spike_idx in range(spike_start,start_end,spike_step):
                    # Long-TFP Part
                    tfp_long = cal_tfp(spike,len(spike[0]) // 2 ,97)
                    tfp_long_sr = sr_net(tfp_long).clip(0,1)
                    # Short-TFP Part
                    tfp = cal_tfp(spike,spike_idx,spike_bsn_len)
                    tfp_bsn = bsn_net(tfp).clip(0,1)
                    tfp_bsn_sr = sr_net(tfp_bsn).clip(0,1)
                    deblur_tea = sdm_deblur(blur,tfp_long_sr,tfp_bsn_sr).clip(0,1)
                    # Deblur_Net Part
                    spike_roi = spike[:,spike_idx - spike_deblur_len // 2:spike_idx + spike_deblur_len // 2 + 1]
                    tfp_long = cal_tfp(spike,len(spike[0]) // 2 ,97)
                    deblur_pred = deblur_net(blur,spike_roi)
                    deblur_pred = deblur_pred.clip(0,1)
                    # Loss
                    if opt.use_ssim:
                        tea_loss += opt.lambda_tea * (1 - loss_ssim(deblur_tea,deblur_pred))  / spike_num
                    tea_loss += opt.lambda_tea * torch.mean(loss_lpips(deblur_tea,deblur_pred) ) / spike_num
                    reblur.append(deblur_pred)
                reblur = torch.mean(torch.stack(reblur,dim = 0),dim = 0,keepdim = False)
                reblur_loss += opt.lambda_reblur * loss_mse(blur,reblur)
                train_loss = tea_loss + reblur_loss
                # optimize
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                scheduler.step()
                writer.add_scalar('Training Loss', train_loss.item())
                # update
                train_loss_all.update(train_loss.item())
                tea_loss_all.update(tea_loss.item())
                reblur_loss_all.update(reblur_loss.item())
            logger.info(f"EPOCH {epoch}/{opt.epochs}: Total Train Loss: {train_loss_all.avg}, Tea Loss: {tea_loss_all.avg},  Reblur Loss: {reblur_loss_all.avg}")
            writer.add_scalar('Epoch Loss', train_loss_all.avg, epoch)
        # visualization result
        if epoch % 5 == 0:
            with torch.no_grad():
                # save the network
                save_network(deblur_net, f"{ckpt_folder}/DeblurNet_{epoch}.pth")
                # visualization
                for batch_idx, (blur,spike,sharp) in enumerate(tqdm(test_loader)):
                    blur,spike = blur.to(device),spike.to(device)
                    spike_idx = len(spike[0]) // 2 
                    # Long-TFP Part
                    tfp_long = cal_tfp(spike,len(spike[0]) // 2 ,97)
                    tfp_long_sr = sr_net(tfp_long).clip(0,1)
                    # Short-TFP Part
                    tfp = cal_tfp(spike,spike_idx,spike_bsn_len)
                    tfp_bsn = bsn_net(tfp).clip(0,1)
                    tfp_bsn_sr = sr_net(tfp_bsn).clip(0,1)
                    deblur_tea = sdm_deblur(blur,tfp_long_sr,tfp_bsn_sr).clip(0,1)
                    # Deblur_Net Part
                    spike_roi = spike[:,spike_idx - spike_deblur_len // 2:spike_idx + spike_deblur_len // 2 + 1]
                    deblur_pred = deblur_net(blur,spike_roi)
                    deblur_pred = deblur_pred.clip(0,1)
                    # visualization
                    if batch_idx in [int(i) for i in np.linspace(0,len(test_loader),5)]:
                        save_img(img = normal_img(deblur_tea[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_tea.png')
                        save_img(img = normal_img(blur[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_blur.png')
                        save_img(img = normal_img(sharp[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_sharp.png')
                        save_img(img = normal_img(deblur_pred[0]),path = f'{img_folder}/{epoch:03}_{batch_idx:04}_deblur.png')

        # save metric result
        if epoch % 50 == 0:
            with torch.no_grad():
                # calculate the metric
                metrics = {}
                method_list = ['SDM','Deblur_Net']
                # metric_list = ['mse','ssim','psnr','lpips']
                metric_list = ['ssim','psnr']
                for method_name in method_list:
                    metrics[method_name] = {}  # 初始化每个方法的字典
                    for metric_name in metric_list:
                        metrics[method_name][metric_name] = AverageMeter()
                for batch_idx, (blur,spike,sharp) in enumerate(tqdm(test_loader)):
                    blur,spike = blur.to(device),spike.to(device)
                    # Long-TFP Part
                    tfp_long = cal_tfp(spike,len(spike[0]) // 2 ,97)
                    tfp_long_sr = sr_net(tfp_long).clip(0,1)
                    # Short-TFP Part
                    tfp = cal_tfp(spike,spike_idx,spike_bsn_len)
                    tfp_bsn = bsn_net(tfp).clip(0,1)
                    tfp_bsn_sr = sr_net(tfp_bsn).clip(0,1)
                    deblur_tea = sdm_deblur(blur,tfp_long_sr,tfp_bsn_sr).clip(0,1)
                    # Deblur_Net Part
                    spike_roi = spike[:,spike_idx - spike_deblur_len // 2:spike_idx + spike_deblur_len // 2 + 1]
                    tfp_long = cal_tfp(spike,len(spike[0]) // 2 ,97)
                    deblur_pred = deblur_net(blur,spike_roi)
                    deblur_pred = deblur_pred.clip(0,1)
                    # Metric
                    for key in metric_list :
                        metrics['SDM'][key].update(compute_img_metric(deblur_tea,sharp,key))
                        metrics['Deblur_Net'][key].update(compute_img_metric(deblur_pred,sharp,key))
                # Print all results
                for method_name in method_list:
                    re_msg = ''
                    for metric_name in metric_list:
                        re_msg += metric_name + ": " + "{:.3f}".format(metrics[method_name][metric_name].avg) + "  "
                    logger.info(f"{method_name}: " + re_msg)
                    writer.add_scalar(f'{method_name}/{metric_name}', metrics[method_name][metric_name].avg, epoch)
    writer.close()
