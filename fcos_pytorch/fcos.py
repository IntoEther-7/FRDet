import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import transforms

from fcos_pytorch.fcos_frdet.fcos import DetectHead, ClipBoxes
from fcos_pytorch.fcos_frdet.loss import GenTargets, LOSS


class FCOSBody(nn.Module):

    def __init__(self, way, shot):
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


class FCOS(nn.Module):

    def __init__(self, way, shot,
                 score_threshold=0.3,
                 nms_iou_threshold=0.2,
                 max_detection_boxes_num=150,
                 mean=None, std=None):
        super().__init__()
        self.strides = [8, 16, 32, 64, 128]
        self.limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

        self.fcos_body = FCOSBody(way, shot)
        self.target_layer = GenTargets(strides=self.strides,
                                       limit_range=self.limit_range)
        self.loss_layer = LOSS()
        self.detection_head = DetectHead(score_threshold=score_threshold,
                                         nms_iou_threshold=nms_iou_threshold,
                                         max_detection_boxes_num=max_detection_boxes_num,
                                         strides=self.strides,
                                         config=None)
        self.clip_boxes = ClipBoxes()
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = [0., 0., 0.]
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = [1., 1., 1.]

    def forward(self, imgs, targets):
        losses = None
        results = None
        if self.training:
            batch_imgs, batch_boxes, batch_classes = collate_fn_train(imgs, targets, self.mean, self.std)
            out = self.fcos_body(batch_imgs)
            targets = self.target_layer(out, batch_boxes, batch_classes)
            losses = self.loss_layer.forward([out, targets])
        else:
            batch_imgs, batch_boxes, batch_classes = collate_fn_val(imgs, targets, self.mean, self.std)
            out = self.fcos_body(batch_imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs, boxes)
            results = (scores, classes, boxes)
        return losses, results


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


if __name__ == '__main__':
    model = FCOSBody(5, 1).cuda()
    print(model)

    input = torch.randn(5, 3, 800, 1024).cuda()
    out = model(input)
    print(out['cls'][0].shape)
    print(out['cnt'][1].shape)
    print(out['reg'][2].shape)
