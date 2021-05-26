from pytorch_exercise.frequency.vgg_gradcam import recovered_net
import torch
from torch._C import device
import torch.nn as nn
import math
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pytorch_exercise.frequency.basicblock import RecoverConv2d
from pytorch_exercise.cnn_cifar10.cam.grad_cam import grad_cam
from pytorch_exercise.cnn_cifar10.model import mysequential
from pytorch_exercise.frequency.vgg_gradcam import recovered_net


class parallel_net(nn.Module):
    def __init__(self, conv_layers, recover_mode = 'W', interpolation = True):
        super(parallel_net, self).__init__()
        print(f'parallel model, recover_mode = {recover_mode}, interpolation = {interpolation}')

        self.recover_backbone = recovered_net(conv_layers, recover_mode, interpolation)
        self.recover_gradcam = recovered_net(conv_layers, recover_mode, interpolation)

        self.optimizer = self.recover_backbone.optimizer
        self.loss = self.recover_backbone.loss
        self.scheduler = self.recover_backbone.scheduler

    def _copy_weight(self):
        with torch.no_grad():
            state_dict = self.recover_backbone.state_dict()
            self.recover_gradcam.load_state_dict(state_dict)

    def forward(self, x, y):
        # update recover_gradcam's parameters from recover_backbone
        self._copy_weight()
        return self.recover_backbone(x)

    