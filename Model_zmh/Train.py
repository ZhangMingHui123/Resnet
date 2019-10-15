from __future__ import print_function, division
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import copy
import types
import cv2
from Model_zmh.model_zmh import Resnet_ZMHfc
from Model_zmh.data_prepare_zmh import DataPrepare
plt.ion()

#====================================Global Variable============================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'E:\DataSet\Resnet\data\hymenoptera_data'
TestFoldDir = "E:\\DataSet\\Resnet\\data\\hymenoptera_data\\val\\ants\\"
checkpoint_dir = "E:\\Studyfile_ZMH\\BBBsjtu\\code\\1.Resnet\\CheckPoint\\1015bestweight.pth"
#====================================Global Variable============================================

Test_transforms=transforms.Compose([
        transforms.ToPILImage(),#将H*W*C的ndarray转成C*H*W的PILImage
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



#Declaration:   训练模型，训练的同时在交叉验证集合上进行测试，保留最好的权重
#input:         model(初始权重为零？随机初始化？), criterion, optimizer, scheduler, num_epochs[default 25]
#output:        训练好的model or 不需要返回值
def train_val_classification(model,criterion, optimizer, scheduler, num_epochs=10):
    BestAcc = 0.0
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch,num_epochs))
        print('-'*20)
        #选择mode,需要在train的同时进行val的验证
        for mode in ['train','val']:
            if mode == 'train':
                model.train()
            else:
                model.eval()
            #loss数据的记录,loss以及分类准确的个数————EpochCorrectNumber
            EpochLoss = 0.0
            EpochCorrectNumber = 0
            for inputs,labels in dataloaders[mode]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(mode =='train'):
                    output = model(inputs)
                    print(output)
                    #都是Tensor张量的形式
                    MaxValues,PredictLabels = torch.max(output,dim=1)
                    loss = criterion(output,labels)
                    if(mode=='train'):
                        loss.backward()
                        optimizer.step()
                #loss.item() -> float number inputs.size[0]->batch
                EpochLoss += loss.item()*inputs.size(0)
                #labels.data可以返回
                EpochCorrectNumber += torch.sum(PredictLabels==labels.data)
            if mode=='train':
                scheduler.step()
            EpochLoss = EpochLoss / dataset_sizes[mode]
            #EpochAcc在0-1之间，必须转为浮点型
            EpochAcc = EpochCorrectNumber.double() / dataset_sizes[mode]
            print('{} Loss:{} ACC:{}'.format(mode,EpochLoss,EpochAcc))
            if EpochAcc > BestAcc and mode == 'val':
                BestAcc = EpochAcc
                print('保存了第{}次结果'.format(epoch+1))
                #torch.save(model.state_dict(),checkpoint_dir)
    print('BestAcc:{}'.format(BestAcc))


#Declaration:   检测模型，简易输出图像，窗口名即为预测的结果
#input:         Model TestFoldDir
#output:        None 或者保存分类好的的图片
def detect_classification(model,TestfoldDir):
    TestFileName = os.listdir(TestfoldDir)
    for i in range(len(TestFileName)):
        TestImageDir = os.path.join(TestfoldDir,TestFileName[i])
        TestImage = TestImage1 = cv2.imread(TestImageDir)
        #TestImage1 = cv2.imread(TestImageDir)
        TestImage = Test_transforms(TestImage)
        TestImage = torch.unsqueeze(TestImage, dim=0)
        TestImage = TestImage.to(device)
        model.load_state_dict(torch.load(checkpoint_dir))
        model.eval()
        with torch.no_grad():
            output = model(TestImage)
            print(output)
            _,pred = torch.max(output,1)
            cv2.imshow('Index:{} predict:{}'.format(i,class_names[pred]),TestImage1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()






if __name__ =='__main__':
    #Load data and Initial model
    dataloaders,dataset_sizes,class_names = DataPrepare(Datadir=data_dir)
    model = Resnet_ZMHfc()
    model = model.to(device)#gpu tensor 加载出来的数据也必须是tensor

    #Set hyper-parameter
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Trian
    # train_val_classification(model=model,criterion=criterion,optimizer=optimizer_ft,scheduler=exp_lr_scheduler)
    # Test
    detect_classification(model=model,TestfoldDir=TestFoldDir)
