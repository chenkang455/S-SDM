from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
from codes.model.networks import *

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


if __name__ == '__main__':
    net = Deblur_Net()
    blur = torch.zeros((1,3,720,1280))
    spike = torch.zeros((1,21,720//4,1280//4))
    flops, params = profile((net).cpu(), inputs=(blur,spike))
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.5fM" % (total/1e6))
    print(net(blur,spike))
