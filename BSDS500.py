import os

import scipy
import numpy as np
from pathlib import Path

from PIL import Image


class BSDS500:

    def __init__(self, dataSetPath: Path):
        self.dataSetPath = dataSetPath
        self.groundTruthPath = Path(dataSetPath, "data", "groundTruth")
        self.imagePath = Path(dataSetPath, "data", "images")
        # lambda x: x[0:-4] remove .mat in file name
        self.trainSet = set(map(lambda x: x[0:-4], os.listdir(Path(self.groundTruthPath, "train"))))
        self.testSet = set(map(lambda x: x[0:-4], os.listdir(Path(self.groundTruthPath, "test"))))
        self.valSet = set(map(lambda x: x[0:-4], os.listdir(Path(self.groundTruthPath, "val"))))

    def get_data(self, set_type, name):
        assert set_type in ['test', 'val', 'train']
        gt = BSDS500._load_ground_truth((Path(self.groundTruthPath, set_type, f"{name}.mat")))
        image = BSDS500._load_image((Path(self.groundTruthPath, set_type, f"{name}.jpg")))
        return image, gt

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
        img = Image.open(path)
        return np.asarray(img)
