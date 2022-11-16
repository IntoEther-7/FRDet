# PROJECT: FRNOD
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-15 13:41
import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

gt_path = "datasets/fsod/annotations/fsod_train.json"  # 存放真实标签的路径
dt_path = "datasets/fsod/annotations/fsod_prediction_r50.json"  # 存放检测结果的路径
# gt_path = 'coco--main/instances_val2017.json'
# dt_path = 'coco--main/my_result_test.json'
# dt_path = "datasets/fsod/annotations/prediction_json_msf/fsod_prediction_msf_mission{}.json".format(i + 1)
# 处理阶段START-----------------------------------
img_ids = []
with open(dt_path, 'r') as f:
    j = json.load(f)
    for obj in tqdm(j, '生成图片列表'):
        if not obj['image_id'] in img_ids:
            img_ids.append(obj['image_id'])
with open(gt_path, 'r') as f:
    j = json.load(f)
    images, type_value, annotations, categories = j.values()
    save_img = []
    for img in tqdm(images, desc='处理图像'):
        if img['id'] in img_ids:
            save_img.append(img)
    save_ann = []
    for ann in tqdm(annotations, desc='处理标注'):
        if ann['image_id'] in img_ids:
            save_ann.append(ann)
    j['images'] = save_img
    j['annotations'] = save_ann
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
os.remove(gt_path)
