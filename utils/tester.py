# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-24 15:23

import json
import os
import sys
from copy import deepcopy
import random

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import transforms
from tqdm import tqdm

from models.FRDet import FRDet

from utils.dataset import CocoDataset


def tester(
        # 基础参数
        way=5, shot=2, query_batch=16, is_cuda=True,
        # 设备参数
        random_seed=None, gpu_index=0,
        # 数据集参数
        root=None, json_path=None, img_path=None,
        # 模型
        model: FRDet = None,
        # 权重文件
        continue_weight=None,
        # 保存相关的参数
        save_root=None
):
    r"""

    :param way:
    :param shot:
    :param query_batch:
    :param is_cuda:
    :param random_seed:
    :param gpu_index:
    :param root:
    :param json_path:
    :param model:
    :param max_iteration:
    :param continue_iteration:
    :param continue_weight: 只写名称即可, 会进入'save_root/weights/'寻找权重文件
    :param save_root:
    :return:
    """
    # 检查参数
    assert root is not None, "root is None"
    assert json_path is not None, "json_path is none"
    assert img_path is not None, "img_path is none"
    # 设置
    torch.set_printoptions(sci_mode=False)
    if random_seed is not None:
        random.seed(random_seed)

    # 设置参数
    torch.cuda.set_device(gpu_index)

    # 生成数据集
    dataset = CocoDataset(root=root, ann_path=json_path, img_path=img_path,
                       way=way, shot=shot, query_batch=query_batch, is_cuda=is_cuda)

    # 模型
    if model is None:
        model = FRDet(
            # box_predictor params
            way, shot, roi_size=7, num_classes=way + 1,
            # backbone
            backbone_name='resnet50', pretrained=True,
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
            box_batch_size_per_image=100, box_positive_fraction=0.25,
            bbox_reg_weights=(10., 10., 5., 5.)
        )

    if is_cuda:
        model.cuda()

    # 保存相关的参数
    save_weights = os.path.join(save_root, 'weights')
    save_results = os.path.join(save_root, 'results')
    save_val = os.path.join(save_root, 'validations')
    save_images = os.path.join(save_val, 'predictions')
    save_json = os.path.join(save_val, 'prediction.json')
    # 创建文件夹保存此次验证
    if not os.path.exists(save_weights):
        os.makedirs(save_weights)
    if not os.path.exists(save_results):
        os.makedirs(save_results)
    if not os.path.exists(save_images):
        os.makedirs(save_images)
    if os.path.exists(save_json):
        print('已经存在预测json, 请检查')

    # 加载权重
    continue_weight = os.path.join(save_root, 'weights', continue_weight)
    weight = torch.load(continue_weight)
    model.load_state_dict(weight['models'])

    # 验证一个轮回
    dataset.initial()
    model.eval()
    predictions = []
    dataset.set_mode(is_training=True)
    predictions.extend(test_iteration(dataset, model, save_images))
    dataset.set_mode(is_training=False)
    predictions.extend(test_iteration(dataset, model, save_images))
    with open(save_json, 'w') as f:
        json.dump(predictions, f)


def test_iteration(dataset, model, save_images):
    pbar = tqdm(dataset)
    predictions = []
    for index, item in enumerate(pbar):
        support, bg, query, query_anns, cat_ids = item
        result = model.forward(support, query, bg, targets=query_anns)
        postfix = {'mission': '{:3}/{:3}'.format(index + 1, len(pbar)),
                   'catIds': cat_ids}
        pbar.set_postfix(postfix)
        predictions.append(result_process(dataset, result, query_anns, save_images))
    return predictions


def result_process(dataset: CocoDataset, result, query_anns, save_images):
    prediction_list = []
    for prediction, gt in zip(result, query_anns):
        image_id = gt['image_id'][0]
        file_name = dataset.coco.loadImgs(image_id)[0]['file_name']
        ori_path = os.path.join(dataset.img_path, file_name)
        save_path = os.path.join(save_images, file_name)
        img = Image.open(ori_path).convert('RGB')
        img_draw = ImageDraw.ImageDraw(img)
        for gt_box in gt['boxes']:
            x1, y1, x2, y2 = gt_box.tolist()
            img_draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline='blue', width=1)
        for bbox, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            bbox = bbox.tolist()
            x1, y1, x2, y2 = bbox
            img_draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red', width=1)
            prediction_list.append({
                "image_id": image_id,
                "bbox": bbox,
                "score": float(score),
                "category_id": int(label)
            })
        save_folder = os.path.split(save_path)[0]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        img.save(save_path)
    return prediction_list
