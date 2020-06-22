import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import torchvision
from models.utils import get_upsampling_weight


class RES32(nn.Module):
    def __init__(self, n_classes=21):
        super(RES32, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        self.conv = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.maxpool.ceil_mode = True
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.score_fr = nn.Conv2d(512, n_classes, 1)
        self.score_fr.weight.data.zero_()
        self.score_fr.bias.data.zero_()
        self.upscore = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(get_upsampling_weight(n_classes, n_classes, 64))

    def forward(self, x):
        x_size = x.size()
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        score_fr = self.score_fr(x)
        upscore = self.upscore(score_fr)
        return upscore[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()


class RES16(nn.Module):
    def __init__(self, n_classes=21):
        super(RES16, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        self.conv = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.maxpool.ceil_mode = True
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.score_layer3 = nn.Conv2d(256, n_classes, kernel_size=1)
        self.score_layer3.weight.data.zero_()
        self.score_layer3.bias.data.zero_()
        self.upscore_layer3 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_layer3.weight.data.copy_(get_upsampling_weight(n_classes, n_classes, 4))

        self.upscore16 = nn.ConvTranspose2d(n_classes, n_classes, 32, 16, bias=False)
        self.upscore16.weight.data.copy_(get_upsampling_weight(n_classes, n_classes, 32))
        self.score_fr = nn.Conv2d(512, n_classes, 1)
        self.score_fr.weight.data.zero_()
        self.score_fr.bias.data.zero_()

    def forward(self, x):
        x_size = x.size()
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        feature4_1 = self.maxpool(x)
        feature4_2 = self.layer1(feature4_1)
        feature8 = self.layer2(feature4_2)
        feature16 = self.layer3(feature8)
        feature32 = self.layer4(feature16)
        score_fr = self.score_fr(feature32)

        layer3_feat16 = self.score_layer3(feature16)
        upscore16 = self.upscore_layer3(score_fr)
        upscore = self.upscore16(
            upscore16[:, :, 1:(1+layer3_feat16.size()[2]), 1:(1+layer3_feat16.size()[3])] + layer3_feat16
        )
        return upscore[:, :, 15: (15 + x_size[2]), 15: (15 + x_size[3])].contiguous()


class RES8(nn.Module):
    def __init__(self, n_classes=21):
        super(RES8, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        self.conv = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.maxpool.ceil_mode = True
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.score_layer3 = nn.Conv2d(256, n_classes, kernel_size=1)
        self.score_layer3.weight.data.zero_()
        self.score_layer3.bias.data.zero_()
        self.upscore_layer3 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_layer3.weight.data.copy_(get_upsampling_weight(n_classes, n_classes, 4))

        self.score_layer2 = nn.Conv2d(128, n_classes, kernel_size=1)
        self.score_layer2.weight.data.zero_()
        self.score_layer2.bias.data.zero_()
        self.upscore_layer2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_layer2.weight.data.copy_(get_upsampling_weight(n_classes, n_classes, 4))

        self.upscore8 = nn.ConvTranspose2d(n_classes, n_classes, 16, 8, bias=False)
        self.upscore8.weight.data.copy_(get_upsampling_weight(n_classes, n_classes, 16))
        self.score_fr = nn.Conv2d(512, n_classes, 1)
        self.score_fr.weight.data.zero_()
        self.score_fr.bias.data.zero_()

    def forward(self, x):
        x_size = x.size()
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        feature4_1 = self.maxpool(x)
        feature4_2 = self.layer1(feature4_1)
        feature8 = self.layer2(feature4_2)
        feature16 = self.layer3(feature8)
        feature32 = self.layer4(feature16)
        score_fr = self.score_fr(feature32)

        layer3_feat16 = self.score_layer3(feature16)
        upscore16 = self.upscore_layer3(score_fr)
        layer2_feat16 = upscore16[:, :, 1:(1+layer3_feat16.size()[2]), 1:(1+layer3_feat16.size()[3])] + layer3_feat16

        layer2_feat8 = self.score_layer2(feature8)
        upscore8 = self.upscore_layer2(layer2_feat16)
        upscore = self.upscore8(
            upscore8[:, :, 1:(1+layer2_feat8.size()[2]), 1:(1+layer2_feat8.size()[3])] + layer2_feat8
        )

        return upscore[:, :, 7: (7 + x_size[2]), 7: (7 + x_size[3])].contiguous()
