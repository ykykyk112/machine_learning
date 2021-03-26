import torch
from torch import nn as nn
from torch import optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

'''
Class Implementation :

    * Modified VGG16 network to apply CAM for input image.
    * Remove FC-layer and add GAP(global average pooling)-layer.
    * Return prediction and feature map of last conv-layer when forwarding.

'''

class vgg_cam(nn.Module):
    def __init__(self):
        super(vgg_cam, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),  # Conv1
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), # Conv2
            nn.MaxPool2d(2, 2),  # Pool1 (56, 56)

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), # Conv3
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), # Conv4
            nn.MaxPool2d(2, 2),  # Pool2 (28, 28)
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), # Conv5
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), # Conv6
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), # Conv7
            nn.MaxPool2d(2, 2),  # Pool3 (14, 14)

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), # Conv8
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), # Conv9
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), # Conv10
            nn.MaxPool2d(2, 2),  # Pool4  (7, 7)

            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), # Conv11
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), # Conv12
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), # Conv13
            #nn.MaxPool2d(2, 2)  # Pool5  (7, 7)
        )
        self.avg_pool = nn.AvgPool2d(7, 7)  # GAP, (512, 1, 1)
        self.classifier = nn.Linear(512, 10)


        self._initialize_weights()

   
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience = 1, factor = 0.1)

    def _initialize_weights(self):
        # call all modules in network (iterable)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for Conv2d-layer
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # weight initialization following normal distribution (mean = 0, std = 1e-2)
                #m.weight.data.normal_(0, 0.01)
                # xavier initialization for fully-connected-layer
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        feature_map = self.features(x)    #   (512, 7, 7)
        out = self.avg_pool(feature_map)    #   (512, 1, 1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, feature_map