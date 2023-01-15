# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 9:54
from collections import OrderedDict
from typing import List, Dict
from torchvision.ops import boxes as box_ops
import torch
from torchvision.models.detection.roi_heads import RoIHeads, maskrcnn_loss, maskrcnn_inference, \
    keypointrcnn_loss, keypointrcnn_inference, fastrcnn_loss

from torch.nn import functional as F
from tqdm import tqdm

from models.FRHead import FRPredictHead, FRPredictHeadWithFlatten
from torchvision.models.detection import _utils as det_utils


class FCOSModifiedRoIHeads(RoIHeads):

    def __init__(self,
                 box_roi_pool,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 ):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = None
        self.mask_head = None
        self.mask_predictor = None

        self.keypoint_roi_pool = None
        self.keypoint_head = None
        self.keypoint_predictor = None

    def forward(self,
                support,
                features,  # type: Dict[str, Tensor]
                proposals,  # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):  # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
                Args:
                    features (List[Tensor])
                    proposals (List[Tensor[N, 4]])
                    image_shapes (List[Tuple[H, W]])
                    targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half, torch.int64)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'

        if self.training:
            # targets, proposals选取训练样本
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        # support align
        if isinstance(support, torch.Tensor):
            support = OrderedDict([('P3', support)])
        device = support['P3'].device
        way_shot, c, h, w = support['P3'].shape
        s_proposals = [torch.Tensor([0, 0, h, w]).unsqueeze(0).to(device) for i in range(way_shot)]
        s_image_shapes = [(h, w) for i in range(way_shot)]
        support = self.box_roi_pool.forward(support, s_proposals, s_image_shapes)
        self.box_roi_pool.scales = None

        # query align
        box_features = self.box_roi_pool.forward(features, proposals, image_shapes)  # 为什么输出为0: 重置scale即可
        self.box_roi_pool.scales = None

        # 送入head
        class_logits, support = self.box_predictor.forward(support, box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier = fcos_modified_focal_fastrcnn_loss(
                class_logits, labels)
            losses = {
                "loss_classifier": loss_classifier
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits,  proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses, support

    def postprocess_detections(self,
                               class_logits,  # type: Tensor
                               proposals,  # type: List[Tensor]
                               image_shapes  # type: List[Tuple[int, int]]
                               ):  # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        pred_scores = F.softmax(class_logits, -1)

        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_scores = []
        all_labels = []
        for scores, image_shape in zip(pred_scores_list, image_shapes):

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            scores, labels = scores[inds], labels[inds]

            all_scores.append(scores)
            all_labels.append(labels)

        return all_scores, all_labels


def modified_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]

    fg_inds = torch.where(labels > 0)[0]
    bg_inds = torch.where(labels == 0)[0]
    tqdm.write('实际背景数/前景数:\t{:>6}/{:<6}, normal_loss'.format(bg_inds.shape[0], fg_inds.shape[0]))

    labels_pos = labels[sampled_pos_inds_subset] - 1
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def modified_focal_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 平衡正负样本
    fg_inds = torch.where(labels > 0)[0]
    bg_inds = torch.where(labels == 0)[0]
    tqdm.write('实际背景数/前景数:\t{:>6}/{:<6}, focal_loss'.format(bg_inds.shape[0], fg_inds.shape[0]))
    fg_loss = F.cross_entropy(class_logits[fg_inds], labels[fg_inds])
    bg_loss = F.cross_entropy(class_logits[bg_inds], labels[bg_inds])
    classification_loss = (fg_loss + bg_loss) / 2

    # 原来的cls_loss
    # classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset] - 1
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def fcos_modified_focal_fastrcnn_loss(class_logits, labels):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)

    # 平衡正负样本
    fg_inds = torch.where(labels > 0)[0]
    bg_inds = torch.where(labels == 0)[0]
    tqdm.write('实际背景数/前景数:\t{:>6}/{:<6}, focal_loss'.format(bg_inds.shape[0], fg_inds.shape[0]))
    fg_loss = F.cross_entropy(class_logits[fg_inds], labels[fg_inds])
    bg_loss = F.cross_entropy(class_logits[bg_inds], labels[bg_inds])
    classification_loss = (fg_loss + bg_loss) / 2

    return classification_loss
