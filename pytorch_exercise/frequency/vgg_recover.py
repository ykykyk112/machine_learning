import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pytorch_exercise.frequency.basicblock import RecoverConv2d

class recovered_net(nn.Module):
    def __init__(self, conv_layers, recover_mode = 'W', interpolation = True):
        super(recovered_net, self).__init__()
        print(f'recover_mode = {recover_mode}, interpolation = {interpolation}')
        self.features = self._make_layer_conv(conv_layers = conv_layers, recover_mode = recover_mode, upsample_mode = interpolation)
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

        self._initialize_weights()
        
        self.basicblock = RecoverConv2d

        # add weight decay(L2), 
        
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)

    def _make_layer_conv(self, conv_layers, recover_mode, upsample_mode):
        
        deconv_mode = upsample_mode
        model = []
        input_size = 3

        for conv in conv_layers:
            if conv == 'R':
                model += [RecoverConv2d(input_size, input_size, kernel_size=3, stride=1, padding = 1, comp_mode = recover_mode, upsample_mode = deconv_mode)]
            elif conv == 'M':
                model += [nn.MaxPool2d(2, 2)]
            else:
                model += [nn.Conv2d(input_size, conv, kernel_size=3, stride=1, padding = 1), 
                          nn.BatchNorm2d(conv),
                          nn.ReLU(inplace=True)]
                input_size= conv
        
        return nn.Sequential(*model)


    def _initialize_weights(self):
        # call all modules in network (iterable)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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
                
                            
    def forward_hook(self, _, inp, out):
        self.feature_maps.append(out)
                            
    def register_hook(self):
        self.feature_maps = []
        self.hook_history = []
        for m in self.modules():
            if isinstance(m, nn.MaxPool2d):
                self.hook_history.append(m.register_forward_hook(self.forward_hook))
    
    def remove_hook(self):
        for h in self.hook_history:
            h.remove()

    def forward(self, x):
        self.feature_maps = []
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x