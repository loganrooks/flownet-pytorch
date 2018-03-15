import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
import torch
import numpy

class Swish(nn.Module):
    def __init__(self, beta=1.0, trainable=False):
        super(Swish, self).__init__()
        self.beta = Variable(torch.cuda.FloatTensor([beta]), requires_grad=trainable)

    def forward(self, x):
        return x * f.sigmoid(self.beta * x)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, batchnorm=True):
        super(Conv, self).__init__()
        padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else Identity()
        self.swish = Swish()

    def forward(self, x):
        conv = self.conv(x)
        norm = self.batchnorm(conv)
        return self.swish(norm)

class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batchnorm=False):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else Identity()
        self.swish = Swish()

    def forward(self, x):
        deconv = self.deconv(x)
        norm = self.batchnorm(deconv)
        return self.swish(norm)