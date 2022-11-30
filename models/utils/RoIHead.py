# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 9:54
from collections import OrderedDict
from typing import List, Dict

import torch
from torchvision.models.detection.roi_heads import RoIHeads, maskrcnn_loss, maskrcnn_inference, \
    keypointrcnn_loss, keypointrcnn_inference, fastrcnn_loss
from torch.nn import functional as F
from torchvision.models.detection import _utils as det_utils

from models.FRHead import FRPredictHead, FRPredictHeadWithFlatten


class ModifiedRoIHeads(RoIHeads):
    def forward(self,
                support,
                bg,
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
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # support align
        device = support.device
        way_shot, c, h, w = support.shape
        support = OrderedDict([('0', support)])
        s_proposals = [torch.Tensor([0, 0, h, w]).unsqueeze(0).to(device) for i in range(way_shot)]
        s_image_shapes = [(h, w) for i in range(way_shot)]
        support = self.box_roi_pool.forward(support, s_proposals, s_image_shapes)
        self.box_roi_pool.scales = None

        # bg align
        device = bg.device
        way_shot, c, h, w = bg.shape
        bg = OrderedDict([('0', bg)])
        bg_proposals = [torch.Tensor([0, 0, h, w]).unsqueeze(0).to(device) for i in range(way_shot)]
        bg_image_shapes = [(h, w) for i in range(way_shot)]
        bg = self.box_roi_pool.forward(bg, bg_proposals, bg_image_shapes)
        self.box_roi_pool.scales = None

        # query align
        box_features = self.box_roi_pool.forward(features, proposals, image_shapes)  # 为什么输出为0: 重置scale即可

        # support, query 送入fc6, fc7
        box_fc = self.box_head(box_features)  # (roi数, 1024?)
        if isinstance(self.box_head, FRPredictHeadWithFlatten):
            support = self.box_head(support)  # (way * shot, 1024)
            bg = self.box_head(bg)  # (way * shot, 1024)

        # 送入head
        class_logits, box_regression, support = self.box_predictor.forward(support, bg, box_features,
                                                                           box_fc)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses, support
