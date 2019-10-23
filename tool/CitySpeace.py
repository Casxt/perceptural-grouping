import os
import random

import cv2
import torch
import numpy as np
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
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
        imgs, gts = self._get_list("train")
        return CitySpaceDataset(list(imgs), list(gts))

    def get_val(self):
        imgs, gts = self._get_list("train")
        return CitySpaceDataset(list(imgs), list(gts))


class CitySpaceDataset(Dataset):
    # std = [0.229, 0.224, 0.225]
    # mean = [0.485, 0.456, 0.406]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    targetTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    randomTransform = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])

    def __init__(self, data: list, groundTruth: list):
        self.data = data
        self.groundTruth = groundTruth

    @staticmethod
    def _load_ground_truth(path: Path):
        return Image.open(path)

    @staticmethod
    def _load_image(path: Path):
        return Image.open(path)

    @staticmethod
    def _get_edge(gt: np.array):
        edge = np.zeros_like(gt).astype(np.float64)
        for i, num in enumerate(np.unique(gt)):
            index = ((gt == num) + 0.)
            sobelx = cv2.Sobel(index, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(index, cv2.CV_64F, 0, 1)
            sobelx = np.uint8(np.absolute(sobelx))
            sobely = np.uint8(np.absolute(sobely))
            sobelcombine = cv2.bitwise_or(sobelx, sobely)
            edge += np.expand_dims(sobelcombine, 0)
        return edge / edge.max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        gt: torch.Tensor = self.targetTransform(
            CitySpaceDataset._load_ground_truth(self.groundTruth[index])
        ).unsqueeze(0)
        image: torch.Tensor = self.transform(CitySpaceDataset._load_image(self.data[index]))
        edge: torch.Tensor = torch.from_numpy(CitySpaceDataset._get_edge(gt.numpy()))
        t = torch.cat((image, gt, edge), 0)
        # remove unused area
        t = t[6:842, 7:2042, :]
        t = t.flip(2) if random.random() < 0.5 else t
        return t[:3], t[3:4], t[4:]
        # return image, gt, edge
