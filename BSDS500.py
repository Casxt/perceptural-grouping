import os
import random

import scipy
import numpy as np
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class BSDS500:

    def __init__(self, dataSetPath: Path):
        self.dataSetPath = dataSetPath
        self.groundTruthPath = Path(dataSetPath, "data", "groundTruth")
        self.imagePath = Path(dataSetPath, "data", "images")
        # lambda x: x[0:-4] remove .mat in file name
        self.trainList = list(map(lambda x: x[0:-4], os.listdir(Path(self.groundTruthPath, "train"))))
        self.testList = list(map(lambda x: x[0:-4], os.listdir(Path(self.groundTruthPath, "test"))))
        self.valList = list(map(lambda x: x[0:-4], os.listdir(Path(self.groundTruthPath, "val"))))

    def get_train(self):
        gts = map(lambda name: (Path(self.groundTruthPath, "train", f"{name}.mat")), self.trainList)
        imgs = map(lambda name: (Path(self.imagePath, "train", f"{name}.jpg")), self.trainList)
        return BSDS500Dataset(list(gts), list(imgs))

    def get_test(self):
        gts = map(lambda name: (Path(self.groundTruthPath, "test", f"{name}.mat")), self.testList)
        imgs = map(lambda name: (Path(self.imagePath, "test", f"{name}.jpg")), self.testList)
        return BSDS500Dataset(list(gts), list(imgs))

    def get_val(self):
        gts = map(lambda name: (Path(self.groundTruthPath, "val", f"{name}.mat")), self.valList)
        imgs = map(lambda name: (Path(self.imagePath, "val", f"{name}.jpg")), self.valList)
        return BSDS500Dataset(list(gts), list(imgs))

    def shuffle(self):
        random.shuffle(self.trainList)
        random.shuffle(self.testList)
        random.shuffle(self.valList)


class BSDS500Dataset(Dataset):
    # std = [0.229, 0.224, 0.225]
    # mean = [0.485, 0.456, 0.406]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    targetTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    def __init__(self, data: list, groundTruth: list):
        self.data = data
        self.groundTruth = groundTruth

    @staticmethod
    def _load_ground_truth(path: Path):
        data = scipy.io.loadmat(path)
        # init boundary to groundTruth shape
        boundary = np.zeros(data["groundTruth"][0, 0]["Boundaries"][0, 0].shape)
        # add all boundary
        for groundTruth in data["groundTruth"][0]:
            boundary += groundTruth["Boundaries"][0, 0]
        # get mean boundary
        boundary /= len(data["groundTruth"][0])
        return boundary

    @staticmethod
    def _load_image(path: Path):
        return Image.open(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        gt = BSDS500Dataset._load_ground_truth(self.groundTruth[index])
        image = BSDS500Dataset._load_image(self.data[index])
        return self.transform(image), self.targetTransform(gt)
