from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18
from einops import repeat



class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, padding=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FCN(nn.Module):
    def __init__(self, img_ch=3, output_ch=11):
        super(FCN, self).__init__()
        self.conv1 = conv_block(img_ch, 32)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(32, 64)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(64, 128)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv_block(128, 256)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = conv_block(256, 256)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, out_channels=output_ch, kernel_size=32, stride=16, padding=8),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(True),
            nn.Conv2d(output_ch, output_ch, kernel_size=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.Maxpool1(x1)

        x2 = self.conv2(x2)
        x3 = self.Maxpool2(x2)

        x3 = self.conv3(x3)
        x4 = self.Maxpool3(x3)

        x4 = self.conv4(x4)
        x5 = self.Maxpool4(x4)
        middle = self.middle(x5)
        out = self.up(middle)

        return out


class FCN_tiny(nn.Module):
    def __init__(self, img_ch=3, output_ch=11):
        super(FCN_tiny, self).__init__()
        self.conv1 = conv_block(img_ch, 32)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(32, 64)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(64, 128)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = conv_block(128, 128)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=8),
            nn.Conv2d(256, output_ch, kernel_size=1)

        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.Maxpool1(x1)

        x2 = self.conv2(x2)
        x3 = self.Maxpool2(x2)

        x3 = self.conv3(x3)
        x4 = self.Maxpool3(x3)

        middle = self.middle(x4)

        

        out = self.up(torch.cat((x4, middle), dim=1))

        return out



import math


def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


import collections



class BasicBlock(nn.Module):
    def __init__(self, in_channel, s):

        super(BasicBlock, self).__init__()
        self.s = s
        self.conv1 = nn.Conv2d(in_channel, in_channel * s, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * s)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel * s, in_channel * s, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel * s)
        if self.s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * s, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(in_channel * s)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 2:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, n_class, zero_init_residual=True):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1),
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channel=128, s=1),
            BasicBlock(in_channel=128, s=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channel=128, s=1),
            BasicBlock(in_channel=128, s=1),
        )

        self.middle = conv_block(128, 128)
        
        
        self.up = nn.Sequential(
            
            

            nn.Upsample(scale_factor=8),
            nn.Conv2d(256, n_class, kernel_size=1)

            
        )

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_ = self.layer3(x)
        x_ = self.layer4(x_)
        middle = self.middle(x_)

        x = self.up(torch.cat((x_, middle), dim=1))
        return x
