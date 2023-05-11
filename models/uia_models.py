import torch
import torch.nn as nn
from models import model_utils as mu


def act_layer(activation_func, inplace=True, neg_slope =0.2, nprelu=1):
    activation_func = activation_func.lower()
    if activation_func == 'relu':
        layer = nn.ReLU(inplace)
    elif activation_func == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif activation_func == 'prelu':
        layer = nn.PReLU(num_parameters=nprelu, init=neg_slope)
    else:
        raise NotImplementedError(f'activation layer {activation_func} is not implemented')
    return layer

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func='relu', mid_channels = None):
        super().__init__()
        if mid_channels == None:    mid_channels = out_channels
        self.activation  = act_layer(activation_func)
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(mid_channels),
            self.activation,
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            self.activation
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func='relu', mid_channels = None):
        super().__init__()
        self.maxpool_dconv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, activation_func, mid_channels)
        )

    def forward(self, x):
        return self.maxpool_dconv(x)

# Decoding block without skip connections
class Up_noskip(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func = 'relu', mid_channles = None, bilinear=True):
        super().__init__()
        if bilinear:
            # scale_factor same as the varible of maxpool()
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            # transposed conv should have the same padding, stride, kernel size as the
            # corresponding down layer in order to maintain the same dimension
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.upnoskip_dconv = DoubleConv(in_channels, out_channels, activation_func, mid_channles)
    
    def forward(self, x):
        x = self.up(x)
        x = self.upnoskip_dconv(x)
        return x
    
# Decoding block with skip connections
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func = 'relu', mid_channles = None, bilinear=True):
        pass

    def forward(self, x):
        pass

class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func='relu'):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, activation_func)
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func = 'relu'):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, activation_func, in_channels)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class SimpleUNet3D(nn.Module):
    def __init__(self, activation_func, in_channels, out_channels = 1):
        super().__init__()
        self.act          = activation_func
        self.in_channels  = in_channels
        self.n1 = in_channels * 2
        self.n2 = self.n1 * 2
        self.n3 = self.n2 * 2
        self.n4 = self.n3 * 2
        # out_channels should be 1. Because we like to output an image,
        # which will be the corrected segmented initial image. 
        self.out_channels = out_channels 

        self.inc   = InConv(self.in_channels, self.n1, self.act)
        self.down1 = Down(self.n1, self.n2, self.act)
        self.down2 = Down(self.n2, self.n3, self.act)
        self.down3 = Down(self.n3, self.n4, self.act)
        self.up1   = Up_noskip(self.n4, self.n3, self.act)
        self.up2   = Up_noskip(self.n3, self.n2, self.act)
        self.up3   = Up_noskip(self.n2, self.n1, self.act)
        self.outc  = OutConv(self.n1, self.out_channels, self.act) 

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.outc(x)
        return x