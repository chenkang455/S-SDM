import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import *
import numpy as np
from codes.model.networks import *
from codes.model.cbam import SpatialGate,ChannelGate,Temporal_Fusion

class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W]
        return x

class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, blind=True,stride=1,padding=0,kernel_size=3):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0,0,1,0))
            self.crop = crop()
        self.replicate = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding,bias=bias) 
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.replicate(x)
        x = self.conv(x)
        x = self.relu(x)        
        if self.blind:
            x = self.crop(x)
        return x

class Pool(nn.Module):
    def __init__(self, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.pool(x)
        return x

class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x

class ENC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, reduce=True, blind=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = Pool(blind=blind)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.reduce:
            x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv4 = Conv(mid_channels, out_channels, bias=bias, blind=blind)

    def forward(self, x, x_in):
        x = self.upsample(x)

        # Smart Padding
        diffY = x_in.size()[2] - x.size()[2]
        diffX = x_in.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Blind_UNet(nn.Module):
    def __init__(self, n_channels=3, n_output=96, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc1 = ENC_Conv(n_channels, 48, 48, bias=bias, blind=blind)
        self.enc2 = ENC_Conv(48, 48, 48, bias=bias, blind=blind)
        self.enc3 = ENC_Conv(48, 96, 48, bias=bias, reduce=False, blind=blind)
        self.dec2 = DEC_Conv(96, 96, 96, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(96+n_channels, 96, n_output, bias=bias, blind=blind)

    def forward(self, input):
        x1 = self.enc1(input)
        x2 = self.enc2(x1)
        x = self.enc3(x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        return x
    
def conv_layer(inDim, outDim, ks, s, p, norm_layer='none'):
    ## convolutional layer
    conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
    relu = nn.ReLU(True)
    assert norm_layer in ('batch', 'instance', 'none')
    if norm_layer == 'none':
        seq = nn.Sequential(*[conv, relu])
    else:
        if (norm_layer == 'instance'):
            norm = nn.InstanceNorm2d(outDim, affine=False, track_running_stats=False) # instance norm
        else:
            momentum = 0.1
            norm = nn.BatchNorm2d(outDim, momentum = momentum, affine=True, track_running_stats=True)
        seq = nn.Sequential(*[conv, norm, relu])
    return seq

def LDI_subNet(inDim=41, outDim=1, norm='none'):  
    ## LDI network
    convBlock1 = conv_layer(inDim,64,3,1,1)
    convBlock2 = conv_layer(64,128,3,1,1,norm)
    convBlock3 = conv_layer(128,64,3,1,1,norm)
    convBlock4 = conv_layer(64,16,3,1,1,norm)
    conv = nn.Conv2d(16, outDim, 3, 1, 1) 
    seq = nn.Sequential(*[convBlock1, convBlock2, convBlock3, convBlock4, conv])
    return seq

class BSN(nn.Module): 
    def __init__(self, n_channels=1, n_output=1, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.n_channels = n_channels
        self.c = n_channels
        self.n_output = n_output
        self.bias = bias
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.unet = Blind_UNet(n_channels=1, bias=bias, blind=blind) 
        self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, n_output, 1, bias=bias)
    
    def forward(self, x):  
        N, C, H, W = x.shape   
        # square
        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')
        x = self.rotate(x)
        x = self.unet(x)   # 4 3 100 100 -> 4 96 100 100
        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x) #4 96 100 100 -> 1 384 100 100
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
import torch
import torch.nn as nn

class MeanShift_GRAY(nn.Conv2d):
    def __init__(
        self, rgb_range, 
        gray_mean=0.446, gray_std=1.0, sign=-1):

        super(MeanShift_GRAY, self).__init__(1, 1, kernel_size=1)
        std = torch.tensor([gray_std])
        self.weight.data = torch.eye(1).view(1, 1, 1, 1) / std.view(1, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor([gray_mean]) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self,color_num = 1, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3 
        scale = 4
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if color_num == 1:
            self.sub_mean = MeanShift_GRAY(1)
            self.add_mean = MeanShift_GRAY(1, sign=1)
        else:
            self.sub_mean = MeanShift(255)
            self.add_mean = MeanShift(255, sign=1)
        
        # define head module
        m_head = [conv(color_num, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, color_num, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 



def conv(inDim,outDim,ks,s,p):
    # inDim,outDim,kernel_size,stride,padding
    conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
    relu = nn.ReLU(True)
    seq = nn.Sequential(conv, relu)
    return seq

def de_conv(inDim,outDim,ks,s,p,op):
    # inDim,outDim,kernel_size,stride,padding
    conv_t = nn.ConvTranspose2d(inDim,outDim, kernel_size=ks, stride=s,
                                padding=p,output_padding= op)
    relu = nn.ReLU(inplace=True)
    seq = nn.Sequential(conv_t, relu)
    return seq

class ChannelAttention(nn.Module):
    ## channel attention block
    def __init__(self, in_planes, ratio=16): 
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    ## spatial attention block
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Deblur_Net(nn.Module):
    def __init__(self,spike_dim = 21) -> None:
        super().__init__()
        # down_sample
        self.blur_enc = conv(3,16,5,2,2)
        self.blur_enc2 = conv(16,32,5,2,2)
        # sample_same
        self.spike_enc = conv(spike_dim,16,3,1,1)
        self.spike_enc2 = conv(16,32,3,1,1)
        # res
        self.resBlock1 = ResnetBlock(64,'zero',get_norm_layer('none'), False, True)
        self.resBlock2 = ResnetBlock(64,'zero',get_norm_layer('none'), False, True)
        # CBAM
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        # up_sample
        self.decoder1 = de_conv(64,32,5,2,2,(1,1))
        self.decoder2 = de_conv(32,16,5,2,2,(1,1))
        self.pred = nn.Conv2d(3 + 16, 3, kernel_size=1, stride=1)
        
    def forward(self,blur,spike):
        # blur branch

        blur_re = self.blur_enc(blur)
        blur_re = self.blur_enc2(blur_re)
        # spike branch
        spike_re = self.spike_enc(spike)
        spike_re = self.spike_enc2(spike_re)
        # fusion
        fusion = torch.cat([blur_re,spike_re],dim = 1)
        fusion = self.ca(fusion) * fusion
        fusion = self.sa(fusion) * fusion
        fusion = self.resBlock1(fusion)
        fusion = self.resBlock2(fusion)
        fusion = self.decoder1(fusion)
        fusion = self.decoder2(fusion)
        result = self.pred(torch.cat([blur,fusion],dim = 1))
        return result
from thop import profile

if __name__ == '__main__':
    # a=DoubleNet().cuda()
    # print(a(torch.ones(2,41,40,40).cuda()))
    net = Deblur_Net()
    # net.load_state_dict(torch.load('exp/Deblur/NEW_GOPRO_9_reblur10_24_full/ckpts/DeblurNet_0.pth'))
    blur = torch.zeros((1,3,720,1280))
    spike = torch.zeros((1,21,720//4,1280//4))
    flops, params = profile((net).cpu(), inputs=(blur,spike))
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.5fM" % (total/1e6))
    print(net(blur,spike))
    # model = EDSR(color_num = 1).cuda()
    # lr_height, lr_width = 90, 180
    # img = torch.ones(1, 1, lr_height, lr_width).cuda() 
    # sr_img = model(img)
    # print(sr_img.shape)
    # model = BSN()
    # arr = torch.ones((1,1,128,128))
    # print(model(arr).shape)
