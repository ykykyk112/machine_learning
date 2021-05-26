import torch
from torch._C import device
import torch.nn as nn
import math
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pytorch_exercise.frequency.basicblock import RecoverConv2d
from pytorch_exercise.cnn_cifar10.cam.grad_cam import grad_cam
from pytorch_exercise.cnn_cifar10.model import mysequential

class recovered_net(nn.Module):
    def __init__(self, conv_layers, recover_mode = 'W', interpolation = True):
        super(recovered_net, self).__init__()
        print(f'recover_mode = {recover_mode}, interpolation = {interpolation}')
        self.features = self._make_layer_conv(conv_layers = conv_layers, recover_mode = recover_mode, upsample_mode = interpolation)
        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

        self._initialize_weights()
        
        # add weight decay(L2), 
        
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-2, momentum = 0.9, weight_decay=0.0015)
        self.loss = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=12, gamma=0.1)

        self.device = torch.device(3)
        self.cam = grad_cam(self)

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
        
        return mysequential.MySequential(*model)


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
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)
        x = self.classifier.forward(x)
        return x

    def forward_cam(self, x, y):
        '''
            get heatmap,
            heritage nn.Sequential and modified forward(x, y) -> only apply y to 'R' layer

        '''
        #cam_ret = self.cam.get_batch_label_cam(x, y)
        cam_ret = self.cam.get_batch_label_cam(x, y)
        cam_ret = cam_ret.to(self.device)
        cam_ret.requires_grad = False
        print(cam_ret.max(), cam_ret.min())
        
        # make optimizer's gradient to zero value, because gradient saved by grad cam operation is dummy gradient.
        self.optimizer.zero_grad()
        x = self.features.forward_cam(x, cam_ret)
        x = x.view(x.size(0), -1)
        x = self.classifier.forward(x)
        return x

    def forward_cam_eval(self, x, y):
        '''
            get heatmap,
            heritage nn.Sequential and modified forward(x, y) -> only apply y to 'R' layer

        '''
        #cam_ret = self.cam.get_batch_label_cam(x, y)
        cam_ret = self.cam.get_cam(x, y)
        #cam_ret = torch.zeros((50, 4, 4))
        cam_ret = cam_ret.to(self.device)


        # make optimizer's gradient to zero value, because gradient saved by grad cam operation is dummy gradient.
        self.optimizer.zero_grad()

        x = self.features.forward_cam(x, cam_ret)
        x = x.view(x.size(0), -1)
        x = self.classifier.forward(x)
        return x

