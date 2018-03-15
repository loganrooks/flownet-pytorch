import torch
import torch.nn as nn
import math
import numpy as np

import layers


class FlowNetS(nn.Module):
    def __init__(self, in_channels=12, batchnorm=True):
        super(FlowNetS, self).__init__()

        self.batchnorm = batchnorm
        self.conv1 = layers.Conv(in_channels, 64, kernel_size=7, stride=2, batchnorm=self.batchnorm)
        self.conv2 = layers.Conv(64, 128, kernel_size=5, stride=2, batchnorm=self.batchnorm)
        self.conv3 = layers.Conv(128, 256, kernel_size=5, stride=2, batchnorm=self.batchnorm)
        self.conv3_1 = layers.Conv(256, 256, kernel_size=3, stride=1, batchnorm=self.batchnorm)
        self.conv4 = layers.Conv(256, 512, kernel_size=3, stride=2, batchnorm=self.batchnorm)
        self.conv4_1 = layers.Conv(512, 512, kernel_size=3, stride=1, batchnorm=self.batchnorm)
        self.conv5 = layers.Conv(512, 512, kernel_size=3, stride=2, batchnorm=self.batchnorm)
        self.conv5_1 = layers.Conv(512, 512, kernel_size=3, stride=1, batchnorm=self.batchnorm)
        self.conv6 = layers.Conv(512, 1024, kernel_size=3, stride=2, batchnorm=self.batchnorm)

        self.deconv5 = layers.Deconv(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv4 = layers.Deconv(1024, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = layers.Deconv(770, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = layers.Deconv(386, 64, kernel_size=4, stride=2, padding=1)

        self.predict_flow5 = self.predict_flow(1024)
        self.predict_flow4 = self.predict_flow(770)
        self.predict_flow3 = self.predict_flow(386)
        self.predict_flow2 = self.predict_flow(194)

        self.upsampled_flow5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    nn.init.uniform(module.bias)
                nn.init.xavier_uniform(module.weight)

            elif isinstance(module, nn.ConvTranspose2d):
                if module.bias is not None:
                    nn.init.uniform(module.bias)
                nn.init.xavier_uniform(module.weight)

    def predict_flow(self, in_channels):
        return nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3_1(self.conv3(conv2_out))
        conv4_out = self.conv4_1(self.conv4(conv3_out))
        conv5_out = self.conv5_1(self.conv5(conv4_out))
        conv6_out = self.conv6(conv5_out)

        deconv5_out = self.deconv5(conv6_out)
        block5 = torch.cat((deconv5_out, conv5_out), 1)
        flow5 = self.predict_flow5(block5)
        flow5_up = self.upsampled_flow5(flow5)

        deconv4_out = self.deconv4(block5)
        block4 = torch.cat((deconv4_out, conv4_out, flow5_up), 1)
        flow4 = self.predict_flow4(block4)
        flow4_up = self.upsampled_flow4(flow4)

        deconv3_out = self.deconv3(block4)
        block3 = torch.cat((deconv3_out, conv3_out, flow4_up), 1)
        flow3 = self.predict_flow3(block3)
        flow3_up = self.upsampled_flow3(flow3)

        deconv2_out = self.deconv2(block3)
        block2 = torch.cat((deconv2_out, conv2_out, flow3_up), 1)
        flow2 = self.predict_flow2(block2)

        if self.training:
            return flow2, flow3, flow4, flow5
        else:
            return flow2


if __name__ == "__main__":
    from torch.autograd import Variable
    input_ = torch.cuda.FloatTensor(np.random.uniform(0, 1, (1, 12, 384, 512)))
    flownet = FlowNetS().cuda()
    flownet.eval()
    output = flownet(Variable(input_))
    print(output.shape)
    print(output)