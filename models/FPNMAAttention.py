# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-12-28 11:57
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from models.utils.MultiplyAttentionModule import MultiplyAttentionModule, PixelAttention, ChannelAttention
import torch
from torch import nn, Tensor
from torchvision.models.detection.image_list import ImageList
from torchvision.transforms import transforms
from torch.nn import functional as F

from models.utils.Inception import Inception


class FPNMAAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, roi_align=None):
        super(FPNMAAttention, self).__init__()
        self.in_channels = in_channels
        self.PA = PixelAttention(in_channels)
        self.CA = ChannelAttention(in_channels, reduction=reduction)
        self.roi_align: MultiScaleRoIAlign = roi_align

    def forward(self, way, shot, support, query, image=None, target=None):
        out = {}
        fg = {}
        loss_attention = None
        for k in query.keys():
            s = support[k]
            _, c, w, h = s.shape
            s = s.reshape(way, shot, c, w, h).mean(1)
            q = query[k]
            channel_attention = self.CA.forward(s)
            q = channel_attention.mean(0).unsqueeze(0) * q
            pixel_attention = F.softmax(self.PA.forward(q), dim=1)
            fg_attention = pixel_attention[:, :1, :, :]
            fg[k] = fg_attention
            out[k] = q * fg_attention
        if self.training:
            mask = self._generate_mask(target, fg, image)
            loss_attention = self._compute_attention_loss(mask, fg)
        return out, loss_attention

    def forward_without_mask(self, way, shot, support, query):
        out = {}
        for k in query.keys():
            s = support[k]
            _, c, w, h = s.shape
            s = s.reshape(way, shot, c, w, h).mean(1)
            q = query[k]
            n, c, w, h = q.shape
            channel_attention = self.CA.forward(s)
            pixel_attention = F.softmax(self.PA.forward(q), dim=1)
            fg_attention = pixel_attention[:, :1, :, :]
            out[k] = (q.unsqueeze(1) * channel_attention.unsqueeze(0) * fg_attention.unsqueeze(1)) \
                .reshape(n, way * c, w, h)
        return out

    def _generate_mask(self, target, fg, image):
        gt_bbox = [i['boxes'] for i in target]
        n, _, w, h = image.tensors.shape
        mask = torch.zeros((n, 1, w, h)).to(image.tensors.device)

        for index, boxes in enumerate(gt_bbox):
            for box in boxes:
                x1, y1, x2, y2 = box
                mask[index, 0, int(y1):int(y2), int(x1):int(x2)] = 1.0

        return mask

    def _compute_attention_loss(self, gt_mask: Tensor, fg):
        loss_attention = 0
        for _fg in fg.values():
            t = transforms.Resize(_fg.shape[2:])
            _mask = t(gt_mask)
            loss_attention += F.binary_cross_entropy(_fg, _mask)
        loss_attention /= len(fg)
        return loss_attention
