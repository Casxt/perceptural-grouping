import os
import random

import cv2
import torch
import numpy as np
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms


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

    def __init__(self, data: list, ground_truth: list, block_size=8):
        self.data = data
        self.ground_truth = ground_truth
        # 获取 获取距离矩阵函数
        self.get_distance_mat = CitySpaceDataset.distance_mat_generator(block_size, block_size)
        self.block_size = block_size

    @staticmethod
    def _load_ground_truth(path: Path):
        i = Image.open(path)
        print(i.size)
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
        dist = np.sqrt(np.power(pos[0] - py, 2) + np.power(pos[1] - px, 2))
        return lambda: dist.clone()

    @staticmethod
    def found_nearest_edge(block: torch.Tensor, get_distance_mat):
        """
        寻找小格中边缘置信度和最邻近中心的边缘点
        @param edge: 1, h, w
        @return: bool, y, x
        """
        c, h, w = block.shape
        # 取出边缘， 边缘值统一为255
        edge_mask = block <= 254
        # 如果小块中没有边缘
        if edge_mask.sum() == edge_mask.shape[0] * edge_mask.shape[1]:
            return False, (w - 1) / 2, (h - 1) / 2
        dist = get_distance_mat()
        dist[edge_mask] = 65535
        ind_max_src = np.unravel_index(np.argmin(dist), dist.shape)
        return True, ind_max_src[0], ind_max_src[1]

    def get_block_ground_truth(self, edge: torch.Tensor):
        """
        获取区块预测的ground truth
        @param edge: 1, h, w
        @return: 3, h/block, w/block, 3个channel分别代表是否包含边缘，边缘y位置， 边缘x位置
        """
        block_size = block_width = block_height = self.block_size
        c, h, w = edge.shape
        block_gt = torch.empty((3, int(h / block_size), int(w / block_size)), dtype=torch.float).to(edge.device)
        for x in range(0, w, block_width):
            for y in range(0, h, block_height):
                block = edge[y:y + block_height, x:x + block_width]
                has_edge, ny, nx = CitySpaceDataset.found_nearest_edge(block, self.get_distance_mat)
                # 归一化 ny nx 到 [-1, 1]
                ny = ny * 2 / block_height - 1
                nx = nx * 2 / block_width - 1
                has_edge = 1 if has_edge else 0
                block_gt[:, y / block_height, x / block_width] = torch.tensor((has_edge, ny, nx), dtype=block_gt.dtype,
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

        edge: torch.Tensor = torch.from_numpy(CitySpaceDataset._get_edge(gt.numpy()[0, :, :])).float()

        return image, gt, edge.unsqueeze(0)
        # return image(3, 600, 800), gt(1, 600, 800), edge(1, 600, 800)
