from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils import data
import os

import matplotlib.pyplot as plt
from PIL import Image
plt.ion()#开启交互模式

#Declaration:   显示dataloader中的图像数据和标签，由于dataloader中的数据是B*C*W*H的 -> W*H*C
#input:         SrcImageFoldDir and DstImageFoldDir
#output:        None
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated

#在线数据处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),#range[0:255] -> [0:1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#output = (output - mean) / std
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]),
}
data_dir = 'E:\DataSet\Resnet\data\hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
#print(class_names[0])


#Declaration:   加载数据,不使用重载Dataloader，直接使用pytorch封装好的模块[transform datasets data]
#input:         Datadir
#output:        ImageDataloader,ImageDatasize,ImageClasses
def DataPrepare(Datadir):
    # 数据增强的格式
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # range[0:255] -> [0:1]
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # output = (output - mean) / std
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    # 数据获取的部分
    ImageDatasets = {x:datasets.ImageFolder(root=os.path.join(Datadir,x),transform=data_transforms[x])
                     for x in ['train','val']}
    ImageDataloader = {x:data.DataLoader(dataset=ImageDatasets[x],batch_size=4,shuffle=True,num_workers=4)
                       for x in ['train','val']}
    ImageDatasize = {x:len(ImageDatasets[x]) for x in ['train','val']}
    ImageClasses  = ImageDatasets['train'].classes
    return ImageDataloader,ImageDatasize,ImageClasses


if __name__ == '__main__':
    for inputs,labels in dataloaders['train']:
        print('InputSize:{}'.format(inputs.size()))
        print('Labels:{}'.format(labels))
        out = torchvision.utils.make_grid(tensor=inputs,padding=2)
        print('OutputSize:{}'.format(out.size()))
        imshow(inp=out,title=[class_names[x] for x in labels])