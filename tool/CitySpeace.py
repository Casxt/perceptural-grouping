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

    def __init__(self, data: list, groundTruth: list):
        self.data = data
        self.groundTruth = groundTruth

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        random.seed(seed)  # apply this seed to img tranfsorms
        gt: torch.Tensor = self.targetTransform(Image.open(self.groundTruth[index]))

        random.seed(seed)
        image: torch.Tensor = self.transform(Image.open(self.data[index]))

        edge: torch.Tensor = torch.from_numpy(CitySpaceDataset._get_edge(gt.numpy()[0, :, :])).float()
        return image, gt, edge.unsqueeze(0)
        # return image, gt, edge

    def __getitem__del(self, index):
        gt: torch.Tensor = self.targetTransform(
            CitySpaceDataset._load_ground_truth(self.groundTruth[index])
        )
        image: torch.Tensor = self.transform(CitySpaceDataset._load_image(self.data[index]))
        edge: torch.Tensor = torch.from_numpy(CitySpaceDataset._get_edge(gt.numpy()[0, :, :])).float()

        t = torch.cat((image, gt.float(), edge.unsqueeze(0)), 0)

        t = t.flip(2) if random.random() < 0.5 else t
        return t[:3], t[3:4], t[4:]
        # return image, gt, edge
