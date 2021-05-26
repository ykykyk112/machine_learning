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

        # register forward & backward hook on last nn.Conv2d module of recover_gradcam
        for m in reversed(list(self.recover_gradcam.modules())):
            if isinstance(m, nn.Conv2d):
                self.hook_history.append(m.register_forward_hook(self.forward_hook))
                self.hook_history.append(m.register_full_backward_hook(self.backward_hook))
                break

    def _copy_weight(self):
        with torch.no_grad():
            state_dict = self.recover_backbone.state_dict()
            self.recover_gradcam.load_state_dict(state_dict)

    def _get_grad_cam(self, x, y):
        self.recover_gradcam.eval()

        output = self.model(x)
        
        loss = 0.
        for idx in range(len(y)):
            loss += output[idx, y[idx]]
        
        loss.backward()

        a_k = torch.mean(self.backward_result, dim=(2, 3), keepdim=True)
        cam = torch.sum(a_k * torch.nn.functional.relu(self.forward_result), dim=1)
        cam_relu = torch.nn.functional.relu(cam)

        return cam_relu

    def forward_hook(self, _, input_image, output):
        self.forward_result = torch.squeeze(output)
        
    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])

    def forward(self, x, y):
        # update recover_gradcam's parameters from recover_backbone
        self._copy_weight()
        # get gradcam heatmap from recover_gradcam model
        heatmap = self._get_grad_cam(x, y)
        print(heatmap.shape)
        return self.recover_backbone(x)

    