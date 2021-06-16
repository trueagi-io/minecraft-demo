import torch
import numpy
from torch import nn

from network import init_weights_xavier
from depth import DepthToSpace


class GoodPoint(nn.Module):
    def __init__(self, grid_size, n_blocks, n_channels=1, activation=nn.ReLU(),
                 batchnorm=True):
        super().__init__()
        self.activation = activation
        stride = 1
        kernel = (3, 3)
        self.n_blocks = n_blocks
        self.conv1a = nn.Conv2d(n_channels, 64, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv2a = nn.Conv2d(64, 64, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv2b = nn.Conv2d(64, 64, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv3a = nn.Conv2d(64, 128, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv3b = nn.Conv2d(128, 128, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv4a = nn.Conv2d(128, 128, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv4b = nn.Conv2d(128, 128, kernel_size=kernel,
                        stride=stride, padding=1)
        self.pool = nn.MaxPool2d((2, 2))

        # Detector head
        self.convPa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(256, 64 * n_blocks, kernel_size=1, stride=1, padding=0)

        if batchnorm:
            self.batchnorm0 = nn.BatchNorm2d(64)
            self.batchnorm1 = nn.BatchNorm2d(64)
            self.batchnorm2 = nn.BatchNorm2d(64)
            self.batchnorm3 = nn.BatchNorm2d(64)
            self.batchnorm4 = nn.BatchNorm2d(128)
            self.batchnorm5 = nn.BatchNorm2d(128)
            self.batchnorm6 = nn.BatchNorm2d(128)
            self.batchnorm7 = nn.BatchNorm2d(128)
            self.batchnormPa = nn.BatchNorm2d(256)
            self.batchnormPb = nn.BatchNorm2d(64 * n_blocks)
        else:
            l = lambda x: x
            self.batchnorm0 = l
            self.batchnorm1 = l
            self.batchnorm2 = l
            self.batchnorm3 = l
            self.batchnorm4 = l
            self.batchnorm5 = l
            self.batchnorm6 = l
            self.batchnorm7 = l
            self.batchnormPb = l
            self.batchnormPa = l
        self.depth_to_space = DepthToSpace(grid_size)
        self.apply(init_weights_xavier)

    def detector_head(self, x):
        # remove batchnorm Pb
        x = self.batchnormPa(self.activation(self.convPa(x)))
        x = self.batchnormPb(self.activation(self.convPb(x)))
        return x

    def superblock(self, x, conv1, conv2, batch1, batch2, skip=False):
        x = conv1(x)
        x = self.activation(x)
        x = batch1(x)
        x = conv2(x)
        x = self.activation(x)
        x = batch2(x)
        return x

    def vgg(self, x):
        x = self.superblock(x, self.conv1a, self.conv1b, self.batchnorm0, self.batchnorm1)
        x = self.pool(x)
        x = self.superblock(x, self.conv2a, self.conv2b, self.batchnorm2, self.batchnorm3)
        x = self.pool(x)
        x = self.superblock(x, self.conv3a, self.conv3b, self.batchnorm4, self.batchnorm5)
        x = self.pool(x)
        x = self.superblock(x, self.conv4a, self.conv4b, self.batchnorm6, self.batchnorm7)
        return x

    def forward(self, x):
        assert x.max() > 0.01
        assert x.max() < 1.01
        x = self.vgg(x)
        semi_det = self.detector_head(x)
        result = self.depth_to_space(semi_det)
        # none_blocks = torch.cat([result[:, 0].unsqueeze(1), result[:, 2:]], dim=1)
        # none_water = torch.cat([result[:, 0].unsqueeze(1), result[:, 1].unsqueeze(1)], dim=1)

        # prob_blocks = nn.functional.softmax(none_blocks, dim=1)
        # prob_water = nn.functional.softmax(none_water, dim=1)
        prob_blocks = nn.functional.softmax(result, dim=1)
        # mean_none = prob_blocks[:, 0] * 0.5 + prob_water[:, 0] * 0.5
        # result = torch.cat([mean_none.unsqueeze(1), prob_water[:, 1].unsqueeze(1), prob_blocks[:, 1:]], dim=1)
        return prob_blocks
