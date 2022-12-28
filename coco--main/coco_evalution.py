# -*- coding: utf-8 -*-
# ======================================================
# @Time    : 2021/02/25
# @Author  : lele wu
# @Email   : 2541612007@qq.com
# @File    : coco_evalution.py
# @Comment: 本脚本用于研究cocoeval.py
# ======================================================
import os.path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pylab, json

if __name__ == "__main__":
    gt_path = "../../FRNOD/datasets/fsod/annotations/fsod_train.json"  # 存放真实标签的路径
    dt_path = os.path.join('../result',
                           'not_flatten_model_20221217_有监督_5x5_FR前景注意力_fsod',
                           'result_fsod_r50_2way_5shot_lr0.002',
                           'validations',
                           'prediction.json')  # 存放检测结果的路径
    # dt_path = "my_result.json"  # 存放检测结果的路径
    print(dt_path)
    # 处理阶段START-----------------------------------
    img_ids = []
    with open(dt_path, 'r') as f:
        j = json.load(f)
        for obj in j:
            if not obj['image_id'] in img_ids:
                img_ids.append(obj['image_id'])
    with open(gt_path, 'r') as f:
        j = json.load(f)
        images, type_value, annotations, categories = j.values()
        for img in images:
            if not img['id'] in img_ids:
                images.remove(img)
        for ann in annotations:
            if not ann['image_id'] in img_ids:
                annotations.remove(ann)
    gt_path = 'tmp.json'
    with open('tmp.json', 'w') as f:
        json.dump(j, f)
    # 处理阶段END--------------------------------------
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
