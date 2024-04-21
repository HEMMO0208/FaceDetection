import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class FaceDataset(Dataset):
    def __init__(self):
        self.label = np.load("../FaceDetect/label.npy")
        self.feat = np.load("../FaceDetect/feat.npy")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = torch.Tensor(self.label[idx])
        img = torch.Tensor(self.feat[idx])

        return img, label
