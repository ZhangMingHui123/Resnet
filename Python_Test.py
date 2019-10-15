import numpy as np
import types
import torchvision
import torch
from torch.autograd import Variable#父类
import sys
import os
cudaAvailable = torch.cuda.is_available()
def to_var(tensor):
    return Variable(tensor.cuda() if cudaAvailable else tensor)
datasets={
    'Train' : 1,
    'Val'   : 2
}
a = torch.tensor([1.0,2.0,3.0])
print(a.requires_grad)
a = to_var(a)
print(a)
print(type(datasets['Train']))
