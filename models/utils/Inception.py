# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-16 15:18
import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.out_channels = 1024
        # 定义分支1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 224, kernel_size=(1, 3), stride=1, padding=1),
            nn.Conv2d(224, 256, kernel_size=(3, 1), stride=1, padding=0)
        )
        # 定义分支2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 192, kernel_size=(5, 1), stride=1, padding=2),
            nn.Conv2d(192, 224, kernel_size=(1, 5), stride=1, padding=0),
            nn.Conv2d(224, 256, kernel_size=(7, 1), stride=1, padding=3),
            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=1, padding=0)
        )
        # 定义分支3
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        )
        # 定义分支4
        self.branch4 = nn.Conv2d(in_channels, 384, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        r"""

        :param x:
        :return: (way, 1024, s1, s2)
        """
        # 计算分支1
        branch1_out = self.branch1(x)
        # 计算分支2
        branch2_out = self.branch2(x)
        # 计算分支3
        branch3_out = self.branch3(x)
        # 计算分支4
        branch4_out = self.branch4(x)

        # 拼接四个不同分支得到的通道，作为输出
        outputs = [branch1_out, branch2_out, branch3_out, branch4_out]
        return torch.cat(outputs, dim=1)  # b,c,w,h  c对应的是dim=1
