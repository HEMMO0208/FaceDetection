import torch
import numpy as np
from train import train_yolo
from models.yolo import YOLO
from models.yolo_loss import yolo_loss
from models.vgg import get_vgg_backbone
from torch.utils.data import TensorDataset


if __name__ == '__main__':
    batch_size = 256
    n_epoch = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.tensor(np.load("feat.npy"), dtype=torch.float)
    y = torch.tensor(np.load("label.npy"), dtype=torch.float)
    dataset = TensorDataset(x, y)
    backbone = get_vgg_backbone()

    yolo_model = YOLO(backbone).to(device)
    optimizer = torch.optim.SGD(yolo_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)

    train_yolo(yolo_model, yolo_loss, optimizer, n_epoch, data_loader, device)