import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

class no_batch_net(nn.Module):
    def __init__(self):
        super(no_batch_net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True),  # Conv1
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True), # Conv2
            nn.MaxPool2d(2, 2),  # Pool1 56

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True), # Conv3
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), # Conv4
#            nn.MaxPool2d(2, 2),  # Pool2 28
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), # Conv5
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True), # Conv6
#            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), # Conv7
            nn.MaxPool2d(2, 2),  # Pool3 14

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True), # Conv8
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True), # Conv9
#            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), # Conv10
#            nn.MaxPool2d(2, 2),  # Pool4 14

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True), # Conv11
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), # Conv12
#            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), # Conv13
            nn.MaxPool2d(2, 2),  # Pool4 7
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 1024), 
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

        self._initialize_weights()

        # add weight decay(L2), 
        
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x