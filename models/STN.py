import copy

from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch
import numpy as np
import itertools
from .tps_grid_gen import TPSGridGen


class block(nn.Module):
    def __init__(self, in_filters, n_filters):
        super(block, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_filters, n_filters, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU())

    def forward(self, x):
        x = self.deconv1(x)
        return x


class block_layerNormal(nn.Module):
    def __init__(self, in_filters, n_filters, shape):
        super(block_layerNormal, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_filters, n_filters, 3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=shape),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.deconv1(x)
        return x





class CNN(nn.Module):
    def __init__(self, input_channels, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(74420, 50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 74420)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x





class simpleCNN_share(nn.Module):
    def __init__(self, input_channels, num_output):
        super(simpleCNN_share, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),

        )
        self.final = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.fc2 = nn.Linear(256, num_output)

    def forward(self, x):

        batch_size = x.size(0)
        features = self.final(self.ConvNet(x)).view(batch_size, -1)
        features = self.localization_fc1(features)
        features = self.fc2(features)

        return features






class BoundedGridLocNet(nn.Module):

    def __init__(self, input_channels, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        
        
        self.cnn = simpleCNN_share(input_channels, grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = torch.tanh(self.cnn(x))
        
        return points.view(batch_size, -1, 2)


class UnBoundedGridLocNet(nn.Module):

    def __init__(self, input_channels, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        
        
        self.cnn = simpleCNN_share(input_channels, grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)


class STNClsNet(nn.Module):

    def __init__(self, input_channels=2):
        super(STNClsNet, self).__init__()
        grid_size = 10
        
        
        
        
        r1 = r2 = 0.9
        
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.0001, 0.2),
            np.arange(-r2, r2 + 0.0001, 0.2),
        )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)

        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }["unbounded_stn"]
        self.loc_net = GridLocNet(input_channels, grid_size, grid_size, target_control_points)

        self.tps = TPSGridGen(256, 256, target_control_points)

    def forward(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, 256, 256, 2)
        
        return grid


class STNClsNet_ori(nn.Module):

    def __init__(self, input_channels=2, shape=256, grid_size=4):
        super(STNClsNet_ori, self).__init__()

        self.shape = shape
        
        
        
        
        r1 = r2 = 0.9
        
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_size - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_size - 1)),
        )))
        
        
        
        
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)

        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }["bounded_stn"]
        self.loc_net = GridLocNet(input_channels, grid_size, grid_size, target_control_points)

        self.tps = TPSGridGen(shape, shape, target_control_points)

    def forward(self, x):
        
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, self.shape, self.shape, 2)
        transformed_x = F.grid_sample(x, grid)
        return transformed_x



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

class SpatialTransformer_FIX(nn.Module):
    def __init__(self, in_channels=1, kernel_size=3, use_dropout=False):
        super(SpatialTransformer_FIX, self).__init__()

        self._in_ch = in_channels
        self._ksize = kernel_size
        self.dropout = use_dropout

        
        self.conv1 = conv_block(in_channels, 64)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(64, 32)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(32, 16)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 6)



    def forward(self, x_):

        shape = x_.shape
        h = shape[2]
        w = shape[3]
        batch_images = x_
        
        x1 = self.conv1(x_)
        x1 = self.max1(x1)

        x2 = self.conv2(x1)
        x2 = self.max2(x2)

        x3 = self.conv3(x2)
        x3 = self.max3(x3)

        x = F.avg_pool2d(x3, kernel_size=x3.size()[2:]).view(x3.size()[0], -1)

        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x)  

        x = x.view(-1, 2, 3)  
        
        
        
        
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, h, w)),align_corners=True)
        assert (affine_grid_points.size(0) == batch_images.size(
            0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points,align_corners=True)
        return rois


