# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-24 15:23

import json
import os
import random

import torch
from PIL import Image, ImageDraw
from PIL import ImageFont
from tqdm import tqdm

from models.FRDet import FRDet
from utils.dataset import CocoDataset

ttf = ImageFont.load_default()


def tester(
        # 基础参数
        way=5, shot=2, query_batch=16, is_cuda=True,
        # 设备参数
        random_seed=None, gpu_index=0,
        # 数据集参数
        root=None, json_path=None, img_path=None, split_cats=None,
        # 模型
        model: FRDet = None,
        # 权重文件
        continue_weight=None,
        # 保存相关的参数
        save_root=None,
        save_json=None
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
                          way=way, shot=shot, query_batch=query_batch, is_cuda=is_cuda, catIds=split_cats)

    # 模型
    if model is None:
        model = FRDet(
            # box_predictor params
            way, shot, roi_size=7, num_classes=way + 1,
            # backbone
            backbone_name='resnet50', pretrained=False,
            returned_layers=None, trainable_layers=4,
            # transform parameters
            min_size=600, max_size=1000,
            image_mean=None, image_std=None,
            # RPN parameters
            rpn_anchor_generator=None, rpn_head=None,
            rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
            rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
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
    if save_json is None:
        save_json = os.path.join(save_val, 'prediction.json')
    else:
        save_json = os.path.join(save_val, save_json)
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
    r"""
    对训练集或测试集, 测试一次
    :param dataset:
    :param model:
    :param save_images:
    :return:
    """
    pbar = tqdm(dataset)
    predictions = []
    for index, item in enumerate(pbar):
        support, bg, query, query_anns, cat_ids = item
        result = model.forward(support, query, bg, targets=query_anns)
        postfix = {'mission': '{:3}/{:3}'.format(index + 1, len(pbar)),
                   'catIds': cat_ids}
        pbar.set_postfix(postfix)
        query_anns, _ = query_anns
        predictions.extend(result_process(dataset, result, query_anns, save_images, cat_ids))
    return predictions


def result_process(dataset: CocoDataset, result, query_anns, save_images, cat_ids):
    r"""
    对结果进行后处理
    :param dataset:
    :param result:
    :param query_anns:
    :param save_images:
    :return:
    """
    prediction_list = []
    for prediction, gt in zip(result, query_anns):
        image_id = gt['image_id'][0]
        file_name = dataset.coco.loadImgs(image_id)[0]['file_name']
        ori_path = os.path.join(dataset.img_path, file_name)
        save_path = os.path.join(save_images, file_name)
        img = Image.open(ori_path).convert('RGB')
        img_draw = ImageDraw.ImageDraw(img)
        for gt_box, gt_labels in zip(gt['boxes'], gt['category_id']):
            x1, y1, x2, y2 = gt_box.tolist()
            img_draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline='blue', width=1)
            img_draw.text((x1, y1), str(gt_labels), font=ttf, fill=(255, 0, 0))
        for bbox, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            bbox = bbox.tolist()
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            img_draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red', width=1)
            img_draw.text((x1, y1), '{:.2f}|{}'.format(float(score), cat_ids[int(label) - 1]), font=ttf,
                          fill=(255, 0, 0))
            prediction_list.append({
                "image_id": image_id,
                "bbox": [x1, y1, w, h],
                "score": float(score),
                "category_id": cat_ids[int(label) - 1]
            })
        save_folder = os.path.split(save_path)[0]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        img.save(save_path)
    return prediction_list
