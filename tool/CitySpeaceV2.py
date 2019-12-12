import os
import random

import cv2
import torch
import numpy as np
from pathlib import Path

from PIL import Image
from collections import namedtuple
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
from typing import List, Any, Callable

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # 遮蔽等级，值越大表示越考靠前

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])


class CitySpace(object):

    def __init__(self, dataSetPath: Path):
        self.dataSetPath = dataSetPath
        self.groundTruthPath = Path(dataSetPath, "data", "groundTruth")
        self.imagePath = Path(dataSetPath, "data", "images")

    def _get_list(self, which: str):
        gt, im = [], []
        for (dirpath, dirnames, filenames) in os.walk(Path(self.dataSetPath, "groundtruth", which)):
            for filename in filenames:
                if "_gtFine_instanceIds.png" in filename:
                    gt.append(Path(dirpath, filename))

        for (dirpath, dirnames, filenames) in os.walk(Path(self.dataSetPath, "leftImg8bit", which)):
            for filename in filenames:
                if ".png" in filename:
                    im.append(Path(dirpath, filename))
        return sorted(im), sorted(gt)

    def get_train(self):
        imgs, gts = self._get_list("train")
        return CitySpaceDataset(list(imgs), list(gts))

    def get_test(self):
        imgs, gts = self._get_list("test")
        return CitySpaceDataset(list(imgs), list(gts))

    def get_val(self):
        imgs, gts = self._get_list("val")
        return CitySpaceDataset(list(imgs), list(gts))


class CitySpaceDataset(Dataset):
    # std = [0.229, 0.224, 0.225]
    # mean = [0.485, 0.456, 0.406]

    transform = transforms.Compose([
        transforms.Lambda(
            lambda img: torchvision.transforms.functional.crop(img, 6, 7, 836, 2035)
        ),
        # transforms.Resize((800, 1600)),
        # transforms.CenterCrop((836, 2035)),
        transforms.RandomCrop((600, 800)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    targetTransform = transforms.Compose([
        transforms.Lambda(
            lambda img: torchvision.transforms.functional.crop(img, 6, 7, 836, 2035)
        ),
        # transforms.Resize((800, 1600)),
        # transforms.CenterCrop((836, 2035)),
        transforms.RandomCrop((600, 800)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    randomTransform = transforms.Compose([
        transforms.CenterCrop((836, 2035)),
        transforms.RandomCrop((400, 600)),
        transforms.RandomHorizontalFlip()
    ])

    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label('unlabeled', 0, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('ego vehicle', 1, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('rectification border', 2, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('out of roi', 3, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('static', 4, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('dynamic', 5, 0, 'void', 0, False, True, (111, 74, 0)),
        Label('ground', 6, 0, 'void', 0, False, True, (81, 0, 81)),
        Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        Label('sidewalk', 8, 0, 'flat', 1, False, False, (244, 35, 232)),
        Label('parking', 9, 0, 'flat', 1, False, True, (250, 170, 160)),
        Label('rail track', 10, 0, 'flat', 1, False, True, (230, 150, 140)),
        Label('building', 11, 0, 'construction', 2, False, False, (70, 70, 70)),
        Label('wall', 12, 1, 'construction', 2, False, False, (102, 102, 156)),
        Label('fence', 13, 1, 'construction', 2, False, False, (190, 153, 153)),
        Label('guard rail', 14, 1, 'construction', 2, False, True, (180, 165, 180)),
        Label('bridge', 15, 1, 'construction', 2, False, True, (150, 100, 100)),
        Label('tunnel', 16, 1, 'construction', 2, False, True, (150, 120, 90)),
        Label('pole', 17, 3, 'object', 3, False, False, (153, 153, 153)),
        Label('polegroup', 18, 3, 'object', 3, False, True, (153, 153, 153)),
        Label('traffic light', 19, 5, 'object', 3, False, False, (250, 170, 30)),
        Label('traffic sign', 20, 5, 'object', 3, False, False, (220, 220, 0)),
        Label('vegetation', 21, 2, 'nature', 4, False, False, (107, 142, 35)),
        Label('terrain', 22, 2, 'nature', 4, False, False, (152, 251, 152)),
        Label('sky', 23, 0, 'sky', 5, False, False, (70, 130, 180)),
        Label('person', 24, 4, 'human', 6, True, False, (220, 20, 60)),
        Label('rider', 25, 4, 'human', 6, True, False, (255, 0, 0)),
        Label('car', 26, 4, 'vehicle', 7, True, False, (0, 0, 142)),
        Label('truck', 27, 4, 'vehicle', 7, True, False, (0, 0, 70)),
        Label('bus', 28, 4, 'vehicle', 7, True, False, (0, 60, 100)),
        Label('caravan', 29, 4, 'vehicle', 7, True, True, (0, 0, 90)),
        Label('trailer', 30, 4, 'vehicle', 7, True, True, (0, 0, 110)),
        Label('train', 31, 4, 'vehicle', 7, True, False, (0, 80, 100)),
        Label('motorcycle', 32, 4, 'vehicle', 7, True, False, (0, 0, 230)),
        Label('bicycle', 33, 4, 'vehicle', 7, True, False, (119, 11, 32)),
        Label('license plate', -1, 0, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    # 各个屏蔽等级中对应的label id
    barrier_label = [
        # 0 级
        {*range(0, 12), 23, -1},
        # 1 级
        {*range(12, 16)},
        # 2 级
        {21, 22},
        # 3 级
        {17, 18},
        # 4 级
        {*range(24, 34)},
        # 5 级
        {*range(24, 34)},
        # 6 级
        {19, 20},
    ]

    def __init__(self, data: list, ground_truth: list, block_size=8):
        self.data = data
        self.ground_truth = ground_truth
        # 获取 获取距离矩阵函数
        self.get_distance_mat = CitySpaceDataset.distance_mat_generator(block_size, block_size)
        self.block_size = block_size

    @staticmethod
    def _load_ground_truth(path: Path):
        i = Image.open(path)
        return i

    @staticmethod
    def _load_image(path: Path):
        return Image.open(path)

    @staticmethod
    def _get_edge(gt: np.array):
        edge = np.zeros_like(gt).astype(np.float64)
        for i, num in enumerate(np.unique(gt)):
            index = (gt == num) + 0.
            sobelx = cv2.Sobel(index, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(index, cv2.CV_64F, 0, 1)
            absX = cv2.convertScaleAbs(sobelx)  # 转回uint8
            absY = cv2.convertScaleAbs(sobely)
            sobelcombine = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            edge += sobelcombine

        too_large = edge > 255
        too_small = edge < 0
        edge[too_large] = 255
        edge[too_small] = 0

        return edge / (edge.max() + 1e-7)

    @staticmethod
    def distance_mat_generator(block_height, block_width):
        """获取坐标矩阵及距离矩阵用于后续运算"""
        dist_source = torch.empty(2, block_height, block_width, dtype=torch.float)
        for y in range(dist_source.shape[2]):
            for x in range(dist_source.shape[1]):
                dist_source[0, y, x], dist_source[1, y, x] = y, x
        px, py = (block_width - 1) / 2, (block_height - 1) / 2
        pos = dist_source
        # 1, 8, 8
        dist = np.sqrt(np.power(pos[0] - py, 2) + np.power(pos[1] - px, 2)).unsqueeze(0)
        return lambda: dist.clone()

    @staticmethod
    def found_nearest_edge(edge_block: torch.Tensor, get_distance_mat):
        """
        小格中边缘置信度  最邻近中心的边缘点 该格所属分类
        @param edge_block: 1, h, w
        @param get_distance_mat: 获取距离矩阵
        @return: bool, y, x, category
        """
        c, h, w = edge_block.shape
        # 取出非边缘， 边缘值会接近1
        no_edge_mask = edge_block < 0.5
        # 如果小块中没有边缘, 不检测其id
        if no_edge_mask.sum() == no_edge_mask.numel():
            return False, (w - 1) / 2, (h - 1) / 2
        dist = get_distance_mat()
        dist[no_edge_mask] = 65535
        ind_max_src = torch.argmin(dist)
        # 行坐标y, 列坐标x, ind_max_src 是Long型，不需要floor
        return True, (ind_max_src / w), (ind_max_src % w)

    @staticmethod
    def get_block_instance(instance_block: torch.Tensor):
        """
        判定小格所属分类，其分类为遮蔽等级最高的instance中像包含素数最多的
        @param instance_block:
        @return:
        """
        # 判定所属分类
        get_id = lambda x: int(x) if x < 1000 else int(x / 1000)
        # 取出所有包含的类型
        block_elem, block_elem_count = torch.unique(instance_block, return_counts=True)
        block_elem, block_elem_count = tuple(block_elem), tuple(block_elem_count)
        # 类型集合, 并将类型转换为int
        block_elem_type = set(map(get_id, block_elem))
        intersection = None
        # 取出该格中最高遮挡级别的分类
        for labels in reversed(CitySpaceDataset.barrier_label):
            intersection = block_elem_type & labels
            if len(intersection) > 0:
                break
        # 取出该格中这些分类的全部实例, 统计各个实例在该格中的像素数
        assert intersection is not None
        label_id = (34, -1)  # id, pixel num
        for i, elem in enumerate(block_elem):
            if get_id(elem) in intersection:
                pixel_num = block_elem_count[i]
                if pixel_num > label_id[1]:
                    label_id = (elem, pixel_num)

        return label_id[0]

    def get_block_ground_truth(self, instance: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """
        获取区块预测的ground truth
        @param instance: 1, h, w
        @param edge: 1, h, w
        @return: 4, h/block, w/block,  4个channel分别代表是否包含边缘，边缘y位置， 边缘x位置, 小块类别
        """
        block_size = block_width = block_height = self.block_size
        c, h, w = edge.shape
        # 4个channel分别代表是否包含边缘，边缘y位置， 边缘x位置, 小块类别
        block_gt = torch.empty((4, int(h / block_size), int(w / block_size)), dtype=torch.float).to(edge.device)
        for x in range(0, w, block_width):
            for y in range(0, h, block_height):
                edge_block = edge[:, y:y + block_height, x:x + block_width]
                instance_block = instance[:, y:y + block_height, x:x + block_width]

                has_edge, ny, nx = CitySpaceDataset.found_nearest_edge(edge_block, self.get_distance_mat)

                # 归一化 ny nx 到 [-1, 1] 对应于 [0, block_width] 或 [0, block_height]
                ny = ny * 2 / block_height - 1
                nx = nx * 2 / block_width - 1
                # block_label = 34  # 34 是数据集中不存在的一个值
                block_label = CitySpaceDataset.get_block_instance(instance_block)
                block_gt[:, int(y / block_height), int(x / block_width)] = \
                    torch.tensor((1 if has_edge else 0, ny, nx, block_label),
                                 dtype=torch.float,
                                 device=block_gt.device)
        return block_gt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        # 固定 seed 保证image 和
        random.seed(seed)  # apply this seed to img tranfsorms
        gt: torch.Tensor = self.targetTransform(Image.open(self.ground_truth[index]))

        random.seed(seed)
        image: torch.Tensor = self.transform(Image.open(self.data[index]))
        # edge 从 0->1, 但并不严格等于0或1
        edge: torch.Tensor = torch.from_numpy(CitySpaceDataset._get_edge(gt.numpy()[0, :, :])).float()
        edge = edge.unsqueeze(0)
        block_gt = self.get_block_ground_truth(gt, edge)
        return image, gt, edge, block_gt
        # return image(3, 600, 800), gt(1, 600, 800), edge(1, 600, 800), block_gt(4, 75, 100)
