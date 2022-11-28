# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-16 14:24
import torch
from torch import nn
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, concat_box_prediction_layers
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops

from models.utils.MultiplyAttentionModule import MultiplyAttentionModule


class MultiplyAttentionRPN(RegionProposalNetwork):
    def __init__(self, way, shot,
                 #
                 anchor_generator,
                 head,
                 #
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0,
                 # MultiplyAttentionModule
                 in_channels=256, reduction=16):
        super(MultiplyAttentionRPN, self).__init__(anchor_generator,
                                                   head,
                                                   fg_iou_thresh,
                                                   bg_iou_thresh,
                                                   batch_size_per_image,
                                                   positive_fraction,
                                                   pre_nms_top_n,
                                                   post_nms_top_n,
                                                   nms_thresh,
                                                   score_thresh=0.0)
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.way = way
        self.shot = shot

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

        self.attention: MultiplyAttentionModule = MultiplyAttentionModule(in_channels, reduction)

    def forward(self,
                support,  # type: List[Tensor]
                images,  # type: ImageList
                features,  # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):  # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        # RPN uses all feature maps that are available
        features = list(features.values())

        # 融合注意力
        features_l = []
        loss_attention = 0
        for feature in features:
            loss_this_level = 0
            c, w, h = feature.shape[1:]
            f_list = []
            for index, f in enumerate(feature):
                f, loss_a = self.attention.forward(support, torch.unsqueeze(f, dim=0), images, targets, index)
                f = f.view(self.way, self.shot, c, w, h)
                f = f.mean(1)
                f = f.view(self.way * c, w, h)
                f_list.append(f)
                if self.training:
                    loss_this_level += loss_a
            loss_attention += loss_this_level / feature.shape[0]
            features_l.append(torch.stack(f_list, dim=0))
        features = features_l
        loss_attention = loss_attention / len(features)

        objectness, pred_bbox_deltas = self.head.forward(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        # 因为每个通道均有, 原始为"s[0] * s[1] * s[2]"
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_attention": loss_attention
            }
        return boxes, losses
