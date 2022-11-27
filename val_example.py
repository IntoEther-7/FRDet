# PROJECT: FRDet
# PRODUCT: PyCharm
# AUTHOR: 17795
# TIME: 2022-11-24 15:22
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

from models.FRDet import FRDet
from utils.tester import tester
from utils.trainer import trainer

torch.set_printoptions(sci_mode=False)

if __name__ == '__main__':
    tester(
        # 基础参数
        way=5, shot=2, query_batch=16, is_cuda=True,
        # 设备参数
        random_seed=114514, gpu_index=0,
        # 数据集参数
        root='../FRNOD/datasets/fsod',
        json_path='annotations/fsod_train.json',
        img_path='images',
        # 模型
        model=None,
        # 权重文件
        continue_weight='FRDet_60000.pth',
        # 保存相关的参数
        save_root='test_val')
