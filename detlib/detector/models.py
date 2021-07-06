#-*- coding: utf-8 -*-

import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

class PNet(nn.Module):
    ''' PNet '''

    def __init__(self, is_train = False):

        super(PNet, self).__init__()

        self.is_train = is_train

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(10),                               # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2),      # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1), # conv2
            nn.PReLU(16),                               # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1), # conv3
            nn.PReLU(32)                                # PReLU3
        )

        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)     # detection
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)     # bbox regresion
        #self.conv4_3 = nn.Conv2d(32, 12, kernel_size=1, stride=1)    # 68-points | 5-points

        self.apply(weights_init)


    def forward(self, x):
        x        = self.pre_layer(x)
        label    = self.softmax4_1(self.conv4_1(x))
        offset   = self.conv4_2(x)
        #landmark = self.conv4_3(x)

        return label, offset


class RNet(nn.Module):
    ''' RNet '''

    def __init__(self,is_train=False):

        super(RNet, self).__init__()

        self.is_train = is_train

        # backbone
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(28),                               # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),      # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1), # conv2
            nn.PReLU(48),                               # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),      # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1), # conv3
            nn.PReLU(64)                                # prelu3

        )

        self.conv4      = nn.Linear(64*2*2, 128)  # conv4
        self.prelu4     = nn.PReLU(128)           # prelu4
        self.conv5_1    = nn.Linear(128, 2)       # detection
        self.softmax5_1 = nn.Softmax(dim=1) 
        self.conv5_2    = nn.Linear(128, 4)       # bounding box regression
        #self.conv5_3    = nn.Linear(128, 12)     # lanbmark localization

        self.apply(weights_init)


    def forward(self, x):

        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)

        det = self.softmax5_1(self.conv5_1(x))
        box = self.conv5_2(x)
        #lmk = self.conv5_3(x)

        return det, box


class ONet(nn.Module):
    ''' RNet '''

    def __init__(self, is_train = False):

        super(ONet, self).__init__()

        self.is_train = is_train

        # backbone
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(32),                               # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),      # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # conv2
            nn.PReLU(64),                               # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),      # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # conv3
            nn.PReLU(64),                               # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2),       # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1),   # conv4
            nn.PReLU(128)                               # prelu4
        )

        self.conv5  = nn.Linear(128*2*2, 256)           # conv5
        self.prelu5 = nn.PReLU(256)                     # prelu5

        self.conv6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.conv6_2 = nn.Linear(256, 4)
        #self.conv6_3 = nn.Linear(256, 12)              # 68 <--> 136; 5 <--> 10

        self.apply(weights_init)

    def forward(self, x):

        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)

        # detection
        det  = self.softmax6_1(self.conv6_1(x))
        box  = self.conv6_2(x)
        #lmk  = self.conv6_3(x)

        return det, box

