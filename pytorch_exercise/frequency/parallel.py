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
    def __init__(self, conv_layers, recover_mode = 'W', interpolation = True, device = None):
        super(parallel_net, self).__init__()
        print(f'parallel model, recover_mode = {recover_mode}, interpolation = {interpolation}')

        self.recover_backbone = recovered_net(conv_layers, recover_mode, interpolation)
        self.recover_gradcam = recovered_net(conv_layers, recover_mode, interpolation)

        self.optimizer = self.recover_backbone.optimizer
        self.loss = self.recover_backbone.loss
        self.scheduler = self.recover_backbone.scheduler

        self.latest_train_cam = torch.ones((50000, 1, 4, 4), dtype=torch.float32, requires_grad=False).to(device)
        self.latest_valid_cam = torch.ones((10000, 1, 4, 4), dtype=torch.float32, requires_grad=False).to(device)
        self.zero_mask = torch.ones((50, 1, 4, 4), dtype=torch.float32, requires_grad=False).to(device)

        # register forward & backward hook on last nn.Conv2d module of recover_gradcam
        for m in reversed(list(self.recover_gradcam.modules())):
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(self.forward_hook)
                m.register_full_backward_hook(self.backward_hook)
                break

    def _copy_weight(self):
        with torch.no_grad():
            state_dict = self.recover_backbone.state_dict()
            self.recover_gradcam.load_state_dict(state_dict)

    def _get_grad_cam(self, x, y, idx, eval):

        self.recover_gradcam.eval()


        # 50 is batch-size
        if not eval:
            latest_heatmap = self.latest_train_cam[idx*50:(idx+1)*50]
        else :
            latest_heatmap = self.latest_valid_cam[idx*50:(idx+1)*50]
        
        output = self.recover_gradcam(x, latest_heatmap)
        
        loss = 0.
        if not eval:
            for i in range(len(y)):
                loss += output[i, y[i]]
        else :
            for i in range(len(y)):
                _, pred = torch.max(output[i], dim = 0)
                loss += output[i, pred]
        
        loss.backward(retain_graph = False)

        self.recover_gradcam.optimizer.zero_grad()

        a_k = torch.mean(self.backward_result, dim=(2, 3), keepdim=True)
        cam = torch.sum(a_k * torch.nn.functional.relu(self.forward_result), dim=1)
        cam_relu = torch.nn.functional.relu(cam).unsqueeze(1).detach()

        if not eval:
            self.latest_train_cam[idx*50:(idx+1)*50] = cam_relu
        else :
            self.latest_valid_cam[idx*50:(idx+1)*50] = cam_relu

        if not eval:
            return cam_relu
        else :
            return self.zero_mask.detach()

    def forward_hook(self, _, input_image, output):
        self.forward_result = torch.squeeze(output)
        
    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])

    def forward(self, x, y, idx, eval = False):
        # update recover_gradcam's parameters from recover_backbone
        self._copy_weight()
        # get gradcam heatmap from recover_gradcam model and update heatmap on self.latest_cam used for forward in _get_grad_cam function
        heatmap = self._get_grad_cam(x, y, idx, eval)
        return self.recover_backbone(x, heatmap)

    