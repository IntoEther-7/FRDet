# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54
import random

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

from models.FRDet import FRDet

torch.set_printoptions(sci_mode=False)

if __name__ == '__main__':
    way = 5
    shot = 2

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    model = FRDet(
        # box_predictor params
        way, shot, roi_size=7, num_classes=way,
        # backbone
        backbone_name='resnet18', pretrained=True,
        returned_layers=None, trainable_layers=3,
        # transform parameters
        min_size=600, max_size=1000,
        image_mean=None, image_std=None,
        # RPN parameters
        rpn_anchor_generator=None, rpn_head=None,
        rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=500,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.3, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=64, box_positive_fraction=0.25,
        bbox_reg_weights=(10., 10., 5., 5.)
    )

    # 测试数据生成
    query_num = 14
    support = [torch.randn([3, 320, 320]).cuda() for i in range(way * shot)]
    query = [torch.randn([3, 1024, 512]).cuda() for i in range(query_num)]
    target = []
    for i in range(query_num):
        n = random.randint(1, 10)
        random_x = random.randint(1, 1024)
        random_y = random.randint(1, 512)
        d = {'boxes': torch.tensor([[random.randint(0, random_x), random.randint(0, random_y),
                                     random.randint(random_x, 1024), random.randint(random_y, 512)] for j in
                                    range(n)]).cuda(),
             'labels': torch.tensor([random.randint(1, 5) for j in range(n)]).cuda()}
        target.append(d)

    # 测试
    model.cuda()
    # --------train
    model.train()
    result = model.forward(support, query, target)
    losses = 0
    for loss in result.values():
        losses += loss
    # --------test
    # model.eval()
    # result = model.forward(support, query, target)
    print()
