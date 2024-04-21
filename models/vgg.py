import torch
import torch.nn as nn


def get_vgg_backbone():
    vgg_net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    for i in range(len(vgg_net.features[:-1])):
        if type(vgg_net.features[i]) == type(nn.Conv2d(64, 64, 3)):
            vgg_net.features[i].weight.requires_grad = False
            vgg_net.features[i].bias.requires_grad = False
            vgg_net.features[i].padding = 1

    return vgg_net.features[:-1]
