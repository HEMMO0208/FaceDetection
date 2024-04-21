import torch.nn as nn
import torch


class YOLO(torch.nn.Module):
    def __init__(self, backbone):
        super(YOLO, self).__init__()
        self.backbone = backbone

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 2048),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(2048, 490)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)

        for m in self.linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        out = self.backbone(x)
        out = self.conv(out)
        out = self.linear(out)
        out = out.view(-1, 7, 7, 10)
        return out
