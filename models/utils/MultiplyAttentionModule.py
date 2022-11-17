# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-16 15:46
import torch
from torch import nn, Tensor
from torchvision.models.detection.image_list import ImageList
from torchvision.transforms import transforms
from torch.nn import functional as F

from models.utils.Inception import Inception


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.inception = Inception(in_channels)
        self.conv = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.inception(x)
        saliency_map = self.conv(x)
        return saliency_map


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(-1, self.in_channels)
        x = F.relu(self.fc1(x), inplace=True)
        channel_attention = torch.sigmoid(self.fc2(x))
        return channel_attention.unsqueeze(2).unsqueeze(3)


class MultiplyAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MultiplyAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.PA = PixelAttention(in_channels)
        self.CA = ChannelAttention(in_channels, reduction=reduction)

    def forward(self, support, query, image=None, target=None, index=None):
        channel_attention = self.CA.forward(support)
        pixel_attention = F.softmax(self.PA.forward(query), dim=1)
        fg_attention = pixel_attention[:, :1, :, :]
        out = query * channel_attention * fg_attention
        if self.training:
            mask = self._generate_mask(image, target, index)
            loss_attention = self._compute_attention_loss(mask, fg_attention)
        else:
            loss_attention = None
        return out, loss_attention

    def _generate_mask(self, image: ImageList, target, index):
        mask = torch.zeros(image.tensors[index].shape[1:]).cuda()
        boxes = target[index]['boxes']
        for box in boxes:
            x1, y1, x2, y2 = box
            mask[int(y1):int(y2), int(x1):int(x2)] = 1.0
        return mask

    def _compute_attention_loss(self, mask: Tensor, fg_attention: Tensor):
        t = transforms.Resize(fg_attention.shape[2:])
        mask = t(mask.unsqueeze(0).unsqueeze(0)).to(fg_attention.device)
        loss_attention = F.binary_cross_entropy(fg_attention, mask)
        return loss_attention
