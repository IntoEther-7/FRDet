# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54
import os.path

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

from models.FRDet import FRDet
from models.FRHead import FRPredictHeadWithFlatten
from utils.tester import tester
from utils.trainer import trainer
from utils.dataset import *

torch.set_printoptions(sci_mode=False)
root = '../FRNOD/datasets/voc/VOCdevkit/VOC2012'
json_path = 'cocoformatJson/voc_2012_train.json'
img_path = 'JPEGImages'
continue_weight = None
save_root = None

def way_shot_test(way, shot, lr, index):
    # result_voc_r50_2way_5shot_lr2e-06_loss_weight_0
    model = FRDet(
        # box_predictor params
        way, shot, roi_size=5, num_classes=way + 1,
        # backbone
        backbone_name='resnet50', pretrained=True,
        returned_layers=None, trainable_layers=3,
        # transform parameters
        min_size=600, max_size=1000,
        image_mean=None, image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=12000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=2000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        rpn_score_thresh=0.5,
        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.7, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=(10., 10., 5., 5.)
    )

    tester(
        # 基础参数
        way=way, shot=shot, query_batch=4, is_cuda=True,
        # 设备参数
        random_seed=None, gpu_index=1,
        # 数据集参数
        root=root,
        json_path=json_path,
        img_path=img_path,
        split_cats=base_ids_voc1,
        # 模型
        model=model,
        # 权重文件
        continue_weight=continue_weight,
        # 保存相关的参数
        save_root=save_root)


if __name__ == '__main__':
    continue_weight = 'FRDet_36_100.pth'
    save_root=os.path.join('result',
                           'not_flatten_model_20230113_fpn_iou严格',
                           'result_voc_r50_5way_5shot_lr0.002')
    way_shot_test(5, 5, 2e-03, 1)
