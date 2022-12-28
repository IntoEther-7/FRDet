# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-17 18:54

from torchvision.models.detection.anchor_utils import AnchorGenerator

from models.FRDet import FRDet
from utils.dataset import *
from utils.tester import tester

torch.set_printoptions(sci_mode=False)
root = '../FRNOD/datasets/fsod'
json_path = 'annotations/fsod_train.json'
img_path = 'images'
continue_weight = 'FRDet_25_2803.pth'
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
        rpn_anchor_generator=AnchorGenerator(((32, 64, 128, 256, 512),), ((0.5, 1.0, 2.0),)),
        rpn_head=None,
        rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=12000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=2000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
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
        random_seed=None, gpu_index=0,
        # 数据集参数
        root=root,
        json_path=json_path,
        img_path=img_path,
        split_cats=None,
        # 模型
        model=model,
        # 权重文件
        continue_weight=continue_weight,
        # 保存相关的参数
        save_root=save_root)


if __name__ == '__main__':
    save_root = os.path.join('result',
                             'not_flatten_model_20221217_有监督_5x5_FR前景注意力_fsod',
                             'result_fsod_r50_2way_5shot_lr0.002')
    way_shot_test(2, 5, 2e-03, 0)
