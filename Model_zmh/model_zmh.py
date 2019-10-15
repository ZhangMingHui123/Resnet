import torch
import torch.nn as nn
import numpy as np
from torchvision import models
class Resnet_ZMHfc(nn.Module):
    def __init__(self):
        super().__init__()
        self.Resnet_layer = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.Resnet_layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x=  self.fc2(x)
        return x


if __name__ == '__main__':
    Resnet_zmh = models.resnet18(pretrained=True)
    print(Resnet_zmh)
    CNN_zmh = Resnet_ZMHfc()
    print(CNN_zmh)