from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from pytorch_exercise.frequency.alexnet import AlexNet
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

        if device != None : 
            self.device = device
 

        self.latest_train_cam = torch.ones((38659, 1, 7, 7), dtype=torch.float32, requires_grad=False).to(self.device)
        self.latest_valid_cam = torch.ones((1500, 1, 7, 7), dtype=torch.float32, requires_grad=False).to(self.device)

        # register forward & backward hook on last nn.Conv2d module of recover_gradcam
        #for m in reversed(list(self.recover_gradcam.modules())):
            #if isinstance(m, RecoverConv2d):
            #if isinstance(m, nn.Conv2d):
                #m.register_forward_hook(self.forward_hook)
                #m.register_full_backward_hook(self.backward_hook)
                #break
        list(self.recover_gradcam.modules())[91].register_forward_hook(self.forward_hook)
        list(self.recover_gradcam.modules())[91].register_full_backward_hook(self.backward_hook)
        print('hook layer :', list(self.recover_gradcam.modules())[91])

    def _copy_weight(self):
        with torch.no_grad():
            state_dict = self.recover_backbone.state_dict()
            self.recover_gradcam.load_state_dict(state_dict)

    def _get_grad_cam(self, x, y, idx, eval):

        self.recover_gradcam.eval()

        batch_size = 32
        train_size, test_size = 38659, 1500
        b_start, b_end = idx*batch_size, (idx+1)*batch_size

        # 50 is batch-size
        if not eval:
            if b_end > train_size : b_end = train_size
            latest_heatmap = self.latest_train_cam[b_start:b_end]
            latest_heatamp = latest_heatmap
        else :
            if b_end > test_size : b_end = test_size
            latest_heatmap = self.latest_valid_cam[b_start:b_end]
            latest_heatamp = latest_heatmap

        output = self.recover_gradcam(x, latest_heatamp)
        
        loss = 0.
        for i in range(len(y)):
            _, pred = torch.max(output[i], dim = 0)
            loss += output[i, pred]

        loss.backward()

        self.recover_gradcam.optimizer.zero_grad()

        if len(self.backward_result.shape) == 3:
            a_k = torch.mean(self.backward_result.unsqueeze(0), dim=(2, 3), keepdim=True)
        else:
            a_k = torch.mean(self.backward_result, dim=(2, 3), keepdim=True)

        cam = torch.sum(a_k * torch.nn.functional.relu(self.forward_result), dim=1)
        cam_relu = torch.nn.functional.relu(cam).unsqueeze(1).detach()

        c_max, c_min = torch.amax(cam_relu, dim = (1, 2, 3)).unsqueeze(1).unsqueeze(1).unsqueeze(1), torch.amin(cam_relu, dim = (1, 2, 3)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        cam_rescaled = (cam_relu - c_min) / ((c_max - c_min)+1e-15)
        

        if not eval:
            if b_end > train_size : b_end = train_size
            self.latest_train_cam[b_start:b_end] = cam_rescaled
        else :
            if b_end > test_size : b_end = test_size
            self.latest_valid_cam[b_start:b_end] = cam_rescaled

        return cam_rescaled

    def forward_hook(self, _, input_image, output):
        self.forward_result = torch.squeeze(output)


    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])

    def activation_hook(self, _, input_image, output):
        self.feature_maps.append(output)

    def register_hook(self):
        self.feature_maps = []
        self.hook_history = []
        for m in self.modules():
            if isinstance(m, nn.MaxPool2d):
                self.hook_history.append(m.register_forward_hook(self.activation_hook))

    
    def forward(self, x, y = None, idx = None, eval = False):
        # update recover_gradcam's parameters from recover_backbone
        self._copy_weight()
        
        # get gradcam heatmap from recover_gradcam model and update heatmap on self.latest_cam used for forward in _get_grad_cam function
        if y == None : heatmap = None
        else : heatmap = self._get_grad_cam(x, y, idx, eval)

        return self.recover_backbone(x, heatmap)

    
