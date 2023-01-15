from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import transforms

from fcos_pytorch.fcos_frdet.fcos import DetectHead, ClipBoxes
from fcos_pytorch.fcos_frdet.loss import GenTargets, LOSS
from models.FPNMAAttention import FPNMAAttention
from models.FRHead import FRPredictHead_FCOS
from utils.FCOSRoIHead import FCOSModifiedRoIHeads


class FCOSBody(nn.Module):

    def __init__(self, way, shot, box_roi_pool):
        super(FCOSBody, self).__init__()
        self.way = way
        self.shot = shot

        self.backbone_with_fpn = resnet_fpn_backbone(backbone_name='resnet50',
                                                     pretrained=True,
                                                     trainable_layers=4,
                                                     returned_layers=None)
        self.conv_out6 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.conv_out7 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.head = FCOSHead(in_channel=256, class_num=self.way)
        self.attention = FPNMAAttention(in_channels=256, reduction=16, roi_align=box_roi_pool)

    def forward(self, x):
        # 特征提取
        fpn = self.backbone_with_fpn.forward(x)

        # 确定P3~P7
        # P2 = fpn['0']
        P3 = fpn['1']
        P4 = fpn['2']
        P5 = fpn['3']
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        features = {
            'P3': P3,
            'P4': P4,
            'P5': P5,
            'P6': P6,
            'P7': P7,
        }
        out = self.head.forward(features)
        return out

    def forward_with_support(self, x, support, targets):
        # 特征提取
        support = self.backbone_with_fpn.forward(support)
        fpn = self.backbone_with_fpn.forward(x)

        # 注意力
        attention_f, loss_attention = self.attention.forward(self.way, self.shot, support, fpn, x, targets)

        # 确定AP3~AP7
        AP3 = attention_f['1']
        AP4 = attention_f['2']
        AP5 = attention_f['3']
        AP6 = self.conv_out6(AP5)
        AP7 = self.conv_out7(F.relu(AP6))
        features = {
            'P3': AP3,
            'P4': AP4,
            'P5': AP5,
            'P6': AP6,
            'P7': AP7,
        }
        out = self.head.forward(features)

        # 确定P3~P7
        query_P3 = fpn['1']
        query_P4 = fpn['2']
        query_P5 = fpn['3']
        query_P6 = self.conv_out6(query_P5)
        query_P7 = self.conv_out7(F.relu(query_P6))
        query_features = {
            'P3': query_P3,
            'P4': query_P4,
            'P5': query_P5,
            'P6': query_P6,
            'P7': query_P7,
        }

        support_P3 = support['1']
        support_P4 = support['2']
        support_P5 = support['3']
        support_P6 = self.conv_out6(support_P5)
        support_P7 = self.conv_out7(F.relu(support_P6))
        support_features = {
            'P3': support_P3,
            'P4': support_P4,
            'P5': support_P5,
            'P6': support_P6,
            'P7': support_P7,
        }
        return {'s': support_features, 'q': query_features}, out, loss_attention


class FCOSHead(nn.Module):

    def __init__(self, in_channel=256, class_num=5):
        super(FCOSHead, self).__init__()
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
        )
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.GroupNorm(32, in_channel),
            nn.ReLU(inplace=True),
        )

        self.cls_branch = nn.Conv2d(in_channel, class_num, kernel_size=(3, 3), padding=(1, 1))
        self.reg_branch = nn.Conv2d(in_channel, 4, kernel_size=(3, 3), padding=(1, 1))
        self.cnt_branch = nn.Conv2d(in_channel, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        cls_logits = {}
        cnt_preds = {}
        reg_preds = {}

        for name, feature in x.items():
            cls_feature = self.cls_conv(feature)
            reg_feature = self.reg_conv(feature)

            cls_logits[name] = self.cls_branch(cls_feature)
            cnt_preds[name] = self.cnt_branch(reg_feature)
            reg_preds[name] = self.reg_branch(reg_feature)

        return {'cls': cls_logits, 'cnt': cnt_preds, 'reg': reg_preds}


class FR_FCOS(nn.Module):

    def __init__(self, way, shot,
                 score_threshold=0.3,
                 nms_iou_threshold=0.2,
                 max_detection_boxes_num=150,
                 roi_size=5,
                 mean=None, std=None):
        super().__init__()
        self.way = way
        self.shot = shot
        self.strides = [8, 16, 32, 64, 128]
        self.limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['P3', 'P4', 'P5', 'P6', 'P7'],
            output_size=roi_size,
            sampling_ratio=2)
        self.fcos_body = FCOSBody(way, shot, box_roi_pool)
        self.target_layer = GenTargets(strides=self.strides,
                                       limit_range=self.limit_range)
        self.loss_layer = LOSS()
        self.detection_head = DetectHead(score_threshold=score_threshold,
                                         nms_iou_threshold=nms_iou_threshold,
                                         max_detection_boxes_num=max_detection_boxes_num,
                                         strides=self.strides,
                                         config=None)
        self.box_roi_pool = box_roi_pool
        self.clip_boxes = ClipBoxes()
        self.fr_head = FRPredictHead_FCOS(way, shot, True)
        self.roi_head = FCOSModifiedRoIHeads(box_roi_pool, self.fr_head, fg_iou_thresh=0.7,
                                             bg_iou_thresh=0.3, batch_size_per_image=150,
                                             positive_fraction=0.7, bbox_reg_weights=None,
                                             score_thresh=0.3, nms_thresh=0.3, detections_per_img=150)
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = [0., 0., 0.]
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = [1., 1., 1.]

    def fcos_forward(self, imgs, targets, support):
        losses = None

        if self.training:
            batch_imgs, batch_boxes, batch_classes = collate_fn_train(imgs, targets, self.mean, self.std)
            support = collate_fn_support(support, self.mean, self.std)
            # out = self.fcos_body.forward(batch_imgs)
            features, out, loss_attention = self.fcos_body.forward_with_support(batch_imgs, support, targets)
            post_targets = self.target_layer(out, batch_boxes, batch_classes)
            losses = self.loss_layer.forward([out, post_targets])
            losses['att_loss'] = loss_attention
        else:
            batch_imgs, batch_boxes, batch_classes = collate_fn_val(imgs, targets, self.mean, self.std)
            support = collate_fn_support(support, self.mean, self.std)
            features, out, _ = self.fcos_body.forward_with_support(batch_imgs, support, targets)

        scores, classes, boxes = self.detection_head(out)
        boxes = self.clip_boxes(batch_imgs, boxes)
        results = {'scores': scores,
                   'classes': classes,
                   'boxes': boxes}
        return losses, results, features

    def forward(self, imgs, targets, support):
        losses, results, features = self.fcos_forward(imgs, targets, support)
        boxes = results['boxes']
        results, losses_cls, support = self.fr_forward(imgs, features, boxes, targets)
        losses.update(losses_cls)
        if self.training:
            loss_aux = self.auxrank(support)
            losses['aux_loss'] = loss_aux
        losses.pop('total_loss')
        return losses, results

    def fr_forward(self, imgs, features, boxes, targets):

        image_shape = [(int(img.shape[1]), int(img.shape[2])) for img in imgs]

        result, losses, support = self.roi_head.forward(features['s'], features['q'],
                                                        [box for box in boxes],
                                                        image_shape, targets=targets)
        return result, losses, support

    def cls_loss(self, fr_scores, targets):
        pass

    def auxrank(self, support: torch.Tensor):
        r"""

        :param support: (way, shot * resolution, channels) -> (way, shot * r, channels)
        :return:
        """
        way = support.size(0)
        shot = support.size(1)
        # (way, shot * resolution, channels) -> (way, shot * r, c)
        support = support / support.norm(2).unsqueeze(-1)
        L1 = torch.zeros((way ** 2 - way) // 2).long().to(support.device)
        L2 = torch.zeros((way ** 2 - way) // 2).long().to(support.device)
        counter = 0
        for i in range(way):
            for j in range(i):
                L1[counter] = i
                L2[counter] = j
                counter += 1
        s1 = support.index_select(0, L1)  # (s^2-s)/2, s, d
        s2 = support.index_select(0, L2)  # (s^2-s)/2, s, d
        dists = s1.matmul(s2.permute(0, 2, 1))  # (s^2-s)/2, s, s
        assert dists.size(-1) == shot
        frobs = dists.pow(2).sum(-1).sum(-1)
        return frobs.sum()


def collate_fn_train(imgs_list, target, mean, std):
    # 处理
    boxes_list = [t['boxes'] for t in target]
    classes_list = [torch.tensor(t['category_id'], dtype=torch.long).to(imgs_list[0].device) for t in target]
    assert len(imgs_list) == len(boxes_list) == len(classes_list)
    batch_size = len(boxes_list)
    pad_imgs_list = []
    pad_boxes_list = []
    pad_classes_list = []

    h_list = [int(s.shape[1]) for s in imgs_list]
    w_list = [int(s.shape[2]) for s in imgs_list]
    max_h = np.array(h_list).max()
    max_w = np.array(w_list).max()
    for i in range(batch_size):
        img = imgs_list[i]
        pad_imgs_list.append(transforms.Normalize(mean, std, inplace=True)(
            torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

    max_num = 0
    for i in range(batch_size):
        n = boxes_list[i].shape[0]
        if n > max_num: max_num = n
    for i in range(batch_size):
        pad_boxes_list.append(
            torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
        pad_classes_list.append(
            torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

    batch_boxes = torch.stack(pad_boxes_list)
    batch_classes = torch.stack(pad_classes_list)
    batch_imgs = torch.stack(pad_imgs_list)

    return batch_imgs, batch_boxes, batch_classes


def collate_fn_val(imgs_list, target, mean, std):
    boxes_list = [t['boxes'] for t in target]
    classes_list = [torch.tensor(t['category_id'], dtype=torch.long).to(imgs_list[0].device) for t in target]
    assert len(imgs_list) == len(boxes_list) == len(classes_list)
    batch_size = len(boxes_list)
    pad_imgs_list = []
    pad_boxes_list = []
    pad_classes_list = []

    h_list = [int(s.shape[1]) for s in imgs_list]
    w_list = [int(s.shape[2]) for s in imgs_list]
    max_h = np.array(h_list).max()
    max_w = np.array(w_list).max()
    for i in range(batch_size):
        img = imgs_list[i]
        pad_imgs_list.append(transforms.Normalize(mean, std, inplace=True)(
            torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

    max_num = 0
    for i in range(batch_size):
        n = boxes_list[i].shape[0]
        if n > max_num: max_num = n
    for i in range(batch_size):
        pad_boxes_list.append(
            torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
        pad_classes_list.append(
            torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

    batch_boxes = torch.stack(pad_boxes_list)
    batch_classes = torch.stack(pad_classes_list)
    batch_imgs = torch.stack(pad_imgs_list)

    return batch_imgs, batch_boxes, batch_classes


def collate_fn_support(imgs_list, mean, std):
    batch_size = len(imgs_list)
    pad_imgs_list = []

    h_list = [int(s.shape[1]) for s in imgs_list]
    w_list = [int(s.shape[2]) for s in imgs_list]
    max_h = np.array(h_list).max()
    max_w = np.array(w_list).max()
    for i in range(batch_size):
        img = imgs_list[i]
        pad_imgs_list.append(transforms.Normalize(mean, std, inplace=True)(
            torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

    batch_imgs = torch.stack(pad_imgs_list)

    return batch_imgs


if __name__ == '__main__':
    model = FCOSBody(5, 1).cuda()
    print(model)

    input = torch.randn(5, 3, 800, 1024).cuda()
    out = model(input)
    print(out['cls'][0].shape)
    print(out['cnt'][1].shape)
    print(out['reg'][2].shape)
