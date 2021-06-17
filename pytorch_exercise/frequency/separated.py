from os import sep
import sys
sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\cnn_cifar10')
sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise')
sys.path.append('C:\\anaconda3\envs\\torch\machine_learning')
from pytorch_exercise.frequency.vgg_gradcam import recovered_net
import torch
from torch._C import device
import torch.nn as nn
import math
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pytorch_exercise.frequency.basicblock import BoundaryConv2d, RecoverConv2d
from pytorch_exercise.cnn_cifar10.cam.grad_cam import grad_cam
from pytorch_exercise.cnn_cifar10.model import mysequential
from pytorch_exercise.frequency.vgg_gradcam import recovered_net

class separated_network(nn.Module):
    def __init__(self, conv_layers, boundary_layers, device):
        super(separated_network, self).__init__()
        self.device = device
        self.features = self._make_layer_conv(conv_layers = conv_layers)
        self.boundary_features, self.compression_conv = self._make_boundary_conv(boundary_layers = boundary_layers)

        for m in self.boundary_features : m = m.to(self.device)
        for m in self.compression_conv : m = m.to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )
        self.boundary_classifier = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Linear(2 * 2 * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

        self.ensemble_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 10)
        )

        self._initialize_weights()
        
        # add weight decay(L2), 
        
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.boundary_loss = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=12, gamma=0.1)


    def _make_layer_conv(self, conv_layers):
        
        model = []
        input_size = 3

        for conv in conv_layers:
            if conv == 'R':
                model += [BoundaryConv2d(input_size, input_size, kernel_size=3, stride=1, padding = 1)]
            elif conv == 'M':
                model += [nn.MaxPool2d(2, 2)]
            else:
                model += [nn.Conv2d(input_size, conv, kernel_size=3, stride=1, padding = 1), 
                          nn.BatchNorm2d(conv),
                          nn.ReLU(inplace = True)]
                input_size= conv
        
        return nn.Sequential(*model)

    def _make_boundary_conv(self, boundary_layers):
        
        model = []
        comp = []

        for conv in boundary_layers:
            model += [nn.Sequential(
                          nn.Conv2d(conv, conv, kernel_size=3, stride=1, padding = 1), 
                          nn.BatchNorm2d(conv),
                          nn.ReLU(inplace = True),
                          nn.AvgPool2d((2, 2)))]
        
        for i in range(len(boundary_layers)-1):
            comp += [nn.Conv2d(boundary_layers[i]+boundary_layers[i+1], boundary_layers[i+1], kernel_size=1, stride=1, padding=0)]

        return model, comp


    def boundary_forward(self):
        x = None
        for idx in range(len(self.boundary_features)):
            if x is None : 
                x = self.boundary_features[idx](self.boundary_maps[idx].to(self.device))
            else :
                x = torch.cat([x, self.boundary_maps[idx].to(self.device)], dim = 1)
                x = self.compression_conv[idx-1](x)
                x = self.boundary_features[idx](x)
        return x

    def _initialize_weights(self):
        # call all modules in network (iterable)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for Conv2d-layer
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # xavier initialization for fully-connected-layer
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def _get_boundary_location(self):
        boundary_maps = []
        for m in self.modules():
            if isinstance(m, BoundaryConv2d):
                boundary_maps.append(m.boundary)
        return boundary_maps

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        self.boundary_maps = self._get_boundary_location()
        b = self.boundary_forward()
        b = b.view(b.size(0), -1)
        b = self.boundary_classifier(b)
        ensemble = self.ensemble_classifier(torch.cat([x, b], dim = 1))
        return x, b, ensemble

