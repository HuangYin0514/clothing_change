# Hybrid Gradient Enhancement Module
import os
import os.path as osp

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


class HGE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(HGE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.fc = nn.Conv2d(out_channels * 5, out_channels, 1, 1, 0, bias=True)
        # 创建一个可训练的卷积核参数，形状为 (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        # 初始化第一行的权重为可训练参数
        nn.init.xavier_uniform_(self.weight[:, :, :, :])

    def forward(self, x):
        # key = 'fc.weight'
        # weights_t = self.state_dict()[key]
        # w1 = weights_t[0, 0, :, :]
        # print(w1)
        # 复制第一行的权重到第三行并取相反数
        b, c, h, w = self.weight.size()
        weight = self.weight.clone()

        Sob0 = weight[:, :, 0, 0]
        Sob45 = weight[:, :, 0, 1]
        Sob90 = weight[:, :, 0, 2]
        Sob135 = weight[:, :, 1, 0]

        GL = weight[:, :, 2, 2]

        weight_0 = torch.zeros(b, c, h, w)

        weight_0[:, :, 0, 0] = Sob0
        weight_0[:, :, 0, 1] = Sob0 * 2
        weight_0[:, :, 0, 2] = Sob0
        weight_0[:, :, 1, 0] = 0
        weight_0[:, :, 1, 1] = 0
        weight_0[:, :, 1, 2] = 0
        weight_0[:, :, 2, 0] = -Sob0
        weight_0[:, :, 2, 1] = -Sob0 * 2
        weight_0[:, :, 2, 2] = -Sob0
        x_0 = F.conv2d(x, weight_0, stride=self.stride, padding=self.padding)

        weight_45 = torch.zeros(b, c, h, w)
        # weight_45=torch.zeros(b,c,h,w)
        weight_45[:, :, 0, 0] = 2 * Sob45
        weight_45[:, :, 0, 1] = Sob45
        weight_45[:, :, 0, 2] = 0
        weight_45[:, :, 1, 0] = Sob45
        weight_45[:, :, 1, 1] = 0
        weight_45[:, :, 1, 2] = -Sob45
        weight_45[:, :, 2, 0] = 0
        weight_45[:, :, 2, 1] = -Sob45
        weight_45[:, :, 2, 2] = -Sob45 * 2
        x_45 = F.conv2d(x, weight_45, stride=self.stride, padding=self.padding)

        weight_90 = torch.zeros(b, c, h, w)
        weight_90[:, :, 0, 0] = Sob90
        weight_90[:, :, 0, 1] = 0
        weight_90[:, :, 0, 2] = -Sob90
        weight_90[:, :, 1, 0] = Sob90 * 2
        weight_90[:, :, 1, 1] = 0
        weight_90[:, :, 1, 2] = -Sob90 * 2
        weight_90[:, :, 2, 0] = Sob90
        weight_90[:, :, 2, 1] = 0
        weight_90[:, :, 2, 2] = -Sob90
        x_90 = F.conv2d(x, weight_90, stride=self.stride, padding=self.padding)

        weight_135 = torch.zeros(b, c, h, w)
        weight_135[:, :, 0, 0] = 0
        weight_135[:, :, 0, 1] = -Sob135
        weight_135[:, :, 0, 2] = -Sob135 * 2
        weight_135[:, :, 1, 0] = Sob135
        weight_135[:, :, 1, 1] = 0
        weight_135[:, :, 1, 2] = -Sob135
        weight_135[:, :, 2, 0] = Sob135 * 2
        weight_135[:, :, 2, 1] = Sob135
        weight_135[:, :, 2, 2] = 0
        x_135 = F.conv2d(x, weight_135, stride=self.stride, padding=self.padding)

        a1 = GL * 4
        a2 = -a1 / 8
        a3 = -a1 / 16

        # kernel_Gaussian_Laplacian=torch.zeros(b,c,5,5)
        kernel_Gaussian_Laplacian = torch.zeros(b, c, 5, 5)
        kernel_Gaussian_Laplacian[:, :, 0, 2] = a3
        kernel_Gaussian_Laplacian[:, :, 1, 1] = a3
        kernel_Gaussian_Laplacian[:, :, 1, 2] = a2
        kernel_Gaussian_Laplacian[:, :, 1, 3] = a3
        kernel_Gaussian_Laplacian[:, :, 2, 0] = a3
        kernel_Gaussian_Laplacian[:, :, 2, 1] = a2
        kernel_Gaussian_Laplacian[:, :, 2, 2] = a1
        kernel_Gaussian_Laplacian[:, :, 2, 3] = a2
        kernel_Gaussian_Laplacian[:, :, 2, 4] = a3
        kernel_Gaussian_Laplacian[:, :, 3, 1] = a3
        kernel_Gaussian_Laplacian[:, :, 3, 2] = a2
        kernel_Gaussian_Laplacian[:, :, 3, 3] = a3
        kernel_Gaussian_Laplacian[:, :, 4, 2] = a3

        x_GL = F.conv2d(x, kernel_Gaussian_Laplacian, stride=self.stride, padding=2)

        x_out = torch.cat((x_0, x_45, x_90, x_135, x_GL), dim=1)
        x_out = self.fc(x_out)

        # 使用F.conv2d进行卷积操作
        return x_out


if __name__ == "__main__":
    inputs = torch.randn(1, 2048, 16, 8)
    hge = HGE(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
    outputs = hge(inputs)
    print(outputs.shape)
