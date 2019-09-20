import os
import random

import torch
import numpy as np
from pathlib import Path

import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class BSDS500:

    def __init__(self, dataSetPath: Path):
        self.dataSetPath = dataSetPath
        self.groundTruthPath = Path(dataSetPath, "data", "groundTruth")
        self.imagePath = Path(dataSetPath, "data", "images")
        # lambda x: x[0:-4] remove .mat in file name
        self.trainList = list(map(lambda x: x[0:-4], os.listdir(Path(self.groundTruthPath, "train")))).sort()
        self.testList = list(map(lambda x: x[0:-4], os.listdir(Path(self.groundTruthPath, "test")))).sort()
        self.valList = list(map(lambda x: x[0:-4], os.listdir(Path(self.groundTruthPath, "val")))).sort()

        # move 50 from val to train, move 150 from test to train

    def get_train(self):
        gts = map(lambda name: (Path(self.groundTruthPath, "train", f"{name}.mat")), self.trainList)
        imgs = map(lambda name: (Path(self.imagePath, "train", f"{name}.jpg")), self.trainList)
        return BSDS500Dataset(list(imgs), list(gts))

    def get_test(self):
        gts = map(lambda name: (Path(self.groundTruthPath, "test", f"{name}.mat")), self.testList)
        imgs = map(lambda name: (Path(self.imagePath, "test", f"{name}.jpg")), self.testList)
        return BSDS500Dataset(list(imgs), list(gts))

    def get_val(self):
        gts = map(lambda name: (Path(self.groundTruthPath, "val", f"{name}.mat")), self.valList)
        imgs = map(lambda name: (Path(self.imagePath, "val", f"{name}.jpg")), self.valList)
        return BSDS500Dataset(list(imgs), list(gts))

    def shuffle(self):
        random.shuffle(self.trainList)
        random.shuffle(self.testList)
        random.shuffle(self.valList)


class BSDS500Dataset(Dataset):
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
        data = sio.loadmat(path)
        # init boundary to groundTruth shape
        boundary = np.zeros(data["groundTruth"][0, 0]["Boundaries"][0, 0].shape)
        # add all boundary
        for groundTruth in data["groundTruth"][0]:
            boundary += groundTruth["Boundaries"][0, 0]
        # get mean boundary
        boundary /= len(data["groundTruth"][0])
        return boundary.astype("float32")

    @staticmethod
    def _load_image(path: Path):
        return Image.open(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        gt = self.targetTransform(BSDS500Dataset._load_ground_truth(self.groundTruth[index]))
        image = self.transform(BSDS500Dataset._load_image(self.data[index]))
        t = torch.cat((image, gt), 0)
        t = t.flip(2) if random.random() < 0.5 else t
        return t[:3], t[3:]
        # return image, gt
