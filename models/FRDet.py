# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-16 10:32
import warnings
from collections import OrderedDict
from typing import List, Tuple

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from models.FRHead import FRBoxHead, FRPredictHead, FRPredictHeadWithFlatten
from models.FeatureExtractor import FeatureExtractor

from models.MARPN import MultiplyAttentionRPN
from models.utils.RoIHead import ModifiedRoIHeads


class FRDet(GeneralizedRCNN):
    def __init__(self,
                 # box_predictor params
                 way, shot, roi_size,
                 num_classes=None,
                 # backbone
                 backbone_name='resnet50', pretrained=False,
                 returned_layers=None, trainable_layers=4,
                 # transform parameters
                 min_size=600, max_size=1000,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=64, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        r"""

        :param way:
        :param shot:
        :param roi_size:
        :param num_classes:
        :param backbone_name:
        :param pretrained:
        :param returned_layers: 默认[3, 4]
        :param trainable_layers:
        :param min_size:
        :param max_size:
        :param image_mean:
        :param image_std:
        :param rpn_anchor_generator:
        :param rpn_head:
        :param rpn_pre_nms_top_n_train:
        :param rpn_pre_nms_top_n_test:
        :param rpn_post_nms_top_n_train:
        :param rpn_post_nms_top_n_test:
        :param rpn_nms_thresh:
        :param rpn_fg_iou_thresh:
        :param rpn_bg_iou_thresh:
        :param rpn_batch_size_per_image:
        :param rpn_positive_fraction:
        :param rpn_score_thresh:
        :param box_roi_pool:
        :param box_head:
        :param box_predictor:
        :param box_score_thresh:
        :param box_nms_thresh:
        :param box_detections_per_img:
        :param box_fg_iou_thresh:
        :param box_bg_iou_thresh:
        :param box_batch_size_per_image:
        :param box_positive_fraction:
        :param bbox_reg_weights:
        """
        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        backbone = FeatureExtractor(backbone_name, pretrained=pretrained, returned_layers=returned_layers,
                                    trainable_layers=trainable_layers)
        out_channels = backbone.out_channels
        channels = out_channels

        # transform
        if image_mean is None:
            image_mean = [0., 0., 0.]
        if image_std is None:
            image_std = [1, 1, 1]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        # RPN
        if rpn_anchor_generator is None:
            anchor_sizes = ((64, 128, 256, 512),)
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels * way, rpn_anchor_generator.num_anchors_per_location()[0]
            )
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn: MultiplyAttentionRPN = MultiplyAttentionRPN(
            way, shot,
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh, in_channels=backbone.out_channels, reduction=16
        )

        # Head
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2)
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = FRBoxHead(
                out_channels * resolution ** 2,
                representation_size)
        if channels < way * resolution:
            Woodubry = False
        else:
            Woodubry = True
        if box_predictor is None:
            representation_size = 1024
            # box_predictor = FRPredictHeadWithFlatten(way, shot, representation_size, num_classes, dropout_rate=0.3)
            box_predictor = FRPredictHead(way, shot, representation_size, num_classes, Woodubry)
        roi_heads = ModifiedRoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        super(FRDet, self).__init__(backbone, rpn, roi_heads, transform)
        self.shot = shot
        self.resolution = roi_size ** 2
        self.roi_size = roi_size
        self.support_transform = GeneralizedRCNNTransform(320, 320, image_mean, image_std)

    def forward(self, support, images, bg, targets=None):
        r"""

        :param support: [tensor(3, w, h)]
        :param images: [tensor(3, w, h)]
        :param targets: [Dict{'boxes': tensor(n, 4), 'labels': tensor(n,)}, 'image_id': int, 'category_id': int, 'id': int]
        :return:
        """
        # 校验及预处理
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        support, _ = self.support_transform(support)
        bg, _ = self.support_transform(bg)
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # 特征提取, 列成字典
        support = self.backbone.forward(support.tensors)  # (way * shot, channels, h, w)
        bg = self.backbone.forward(bg.tensors)  # (way * shot, channels, h, w)
        features = self.backbone.forward(images.tensors)  # (n, channels, h, w)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(support, images, features, targets)
        detections, detector_losses, support = self.roi_heads.forward(support, bg, features, proposals,
                                                                      images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        aux_loss = {}
        if self.training:
            aux_loss = self.auxrank(support)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update({'loss_aux': aux_loss})

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

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
        return frobs.sum().mul(0.1)


if __name__ == '__main__':
    support = [torch.randn([3, 320, 320]).cuda() for i in range(10)]
    query = [torch.randn([3, 1024, 512]).cuda() for i in range(5)]
    model = FRDet(way=5, shot=2, roi_size=7, num_classes=5).cuda()
    model.eval()
    result = model.forward(support, query)
