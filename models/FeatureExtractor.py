# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-16 10:45
import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from models.utils.Inception import Inception


class FeatureExtractor(nn.Module):

    def __init__(self,
                 # fpn backbone
                 backbone_name='resnet50', pretrained=True, returned_layers=None, trainable_layers=3):
        r"""

        :param backbone_name: 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        :param pretrained: 是否预训练
        :param returned_layers: 返回那一层, [1, 4], 对应着c1-c4层, 返回
        :param trainable_layers: 训练哪几层, 从最后一层往前数
        """
        super(FeatureExtractor, self).__init__()
        if returned_layers is None:
            returned_layers = [3, 4]
        self.out_channels = 256
        self.s_scale = 16

        self.backbone = resnet_fpn_backbone(backbone_name, pretrained=pretrained, trainable_layers=trainable_layers,
                                            returned_layers=returned_layers)  # (n, 256, x, x)
        self.inception = Inception(256)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2.)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1280, 1024, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(inplace=True)
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        r"""
        进行特征提取
        :param x_list: List[Tensor]
        :return: List[Tensor]
        """
        features = self.backbone.forward(x)
        c3 = features['0']  # 缩小16倍
        c3 = self.inception.forward(c3)
        c4 = features['1']  # 缩小32倍
        c4 = self.upsample(c4)  # 缩小16倍
        out = torch.cat([c3, c4], dim=1)
        out = self.conv(out)
        return out


class FeatureExtractorOnly(nn.Module):

    def __init__(self,
                 # fpn backbone
                 backbone_name='resnet50', pretrained=True, returned_layers=None, trainable_layers=3):
        r"""

        :param backbone_name: 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        :param pretrained: 是否预训练
        :param returned_layers: 返回那一层, [1, 4], 对应着c1-c4层, 返回
        :param trainable_layers: 训练哪几层, 从最后一层往前数
        """
        super(FeatureExtractorOnly, self).__init__()
        if returned_layers is None:
            returned_layers = [1, 2, 3, 4]
        self.out_channels = 256

        self.backbone = resnet_fpn_backbone(backbone_name, pretrained=pretrained, trainable_layers=trainable_layers,
                                            returned_layers=returned_layers)  # (n, 256, x, x)
        # self.inception = Inception(256)
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2.)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1280, 1024, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(inplace=True)
        # )
        # self.conv = nn.Sequential(
        #     nn.Conv2d(3072, 1024, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(inplace=True)
        # )

    def forward(self, x):
        r"""
        进行特征提取
        :param x_list: List[Tensor]
        :return: List[Tensor]
        """
        features = self.backbone.forward(x)
        # out = features['0']  # 缩小8倍
        # c2 = self.inception.forward(c2)
        # c3 = features['1']  # 缩小16倍
        # c3 = self.upsample(c3)  # 缩小16倍
        # c3 = self.inception.forward(c3)
        # c4 = features['2']  # 缩小32倍
        # c4 = self.upsample(c4)  # 缩小16倍
        # c4 = self.upsample(c4)  # 缩小8倍
        # c4 = self.inception.forward(c4)
        # out = torch.cat([c2, c3, c4], dim=1)
        # out = self.conv(out)
        return features


if __name__ == '__main__':
    x = torch.randn([5, 3, 1024, 512])
    resnet50 = FeatureExtractorOnly(backbone_name='resnet50', pretrained=True)
    result = resnet50.forward(x)
    print()
