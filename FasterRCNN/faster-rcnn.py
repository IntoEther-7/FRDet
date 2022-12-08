# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-12-08 15:32
import json
import os

import torch
from PIL import ImageDraw, ImageFont, Image
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from utils.dataset import CocoDataset

torch.set_printoptions(sci_mode=False)
root = '../../FRNOD/datasets/coco'
json_path = 'annotations/instances_train2017.json'
img_path = 'train2017'


def tester_faster_rcnn(
        # 基础参数
        way=5, shot=2, query_batch=16, is_cuda=True,
        # 设备参数
        random_seed=None, gpu_index=0,
        # 数据集参数
        root=None, json_path=None, img_path=None,
):
    save_root = '../result/old_model/result_coco_r50_{}way_{}shot'.format(way, shot)
    model = fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)
    torch.cuda.set_device(gpu_index)
    model.cuda()
    model.eval()

    dataset = CocoDataset(root=root, ann_path=json_path, img_path=img_path,
                          way=way, shot=shot, query_batch=query_batch, is_cuda=is_cuda)

    # 创建文件夹保存此次验证
    save_weights = os.path.join(save_root, 'weights')
    save_results = os.path.join(save_root, 'results')
    save_val = os.path.join(save_root, 'validations')
    save_images = os.path.join(save_val, 'predictions')
    save_json = os.path.join(save_val, 'prediction.json')

    if not os.path.exists(save_weights):
        os.makedirs(save_weights)
    if not os.path.exists(save_results):
        os.makedirs(save_results)
    if not os.path.exists(save_images):
        os.makedirs(save_images)
    if os.path.exists(save_json):
        print('已经存在预测json, 请检查')

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
        result = model.forward(query, targets=query_anns)
        postfix = {'mission': '{:3}/{:3}'.format(index + 1, len(pbar)),
                   'catIds': cat_ids}
        pbar.set_postfix(postfix)
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
    ttf = ImageFont.load_default()
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
            img_draw.text((x1, y1), '{:.2f}|{}'.format(float(score), label), font=ttf,
                          fill=(255, 0, 0))
            prediction_list.append({
                "image_id": image_id,
                "bbox": [x1, y1, w, h],
                "score": float(score),
                "category_id": label
            })
        save_folder = os.path.split(save_path)[0]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        img.save(save_path)
    return prediction_list


if __name__ == '__main__':
    way = 5
    shot = 5

    tester_faster_rcnn(way, shot, root=root, json_path=json_path, img_path=img_path)
