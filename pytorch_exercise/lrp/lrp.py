import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LRP():
    def __init__(self, model, module_list, device):
        self.model = model
        self.device = device
        self.gamma = 0.
        self.epsilon = 0.
        self.module_list = tuple(module_list)
        self.model_layers = []
        self.model_activation = []
        self.hook_history = []
        # get all networks's layers
        self.set_layer()
        self.register_hook()
  
    # Initializing network's modules, which used in backward propagation(nn.Linear or nn.Conv2d or nn.AvgPool2d)
    def set_layer(self):
        module = list(self.model.modules())
        module.reverse()
        for m in module:
            if isinstance(m, self.module_list):
                if isinstance(m, nn.Linear):
                    m.bias.data.zero_()
                if isinstance(m, nn.Conv2d):
                    m.bias.data.zero_()
                if isinstance(m, nn.MaxPool2d):
                    m = nn.AvgPool2d(2, 2)
                self.model_layers.append(m)
    
    # Forward hook to get activation value after each ReLU function
    def forward_hook(self, _, image, output):
        self.model_activation.append(output)
    
    def register_hook(self) :
        for m in self.model.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d):
                hook = m.register_forward_hook(self.forward_hook)
                self.hook_history.append(hook)
                
    def remove_hook(self) :
        for h in self.hook_history:
            h.remove()
        self.hook_history = []
        
    def forward(self, x):
        with torch.no_grad():
            self.model.eval()
            self.model_activation = []
            self.model_activation.append(x)
            ret = self.model.forward(x).data
            self.model_activation.reverse()
        return ret
                
    def rho(self, layer, gamma) :
        with torch.no_grad():
            if gamma != 0. and not isinstance(layer, nn.AvgPool2d):
                l = copy.deepcopy(layer)
                pos_idx = torch.where(l.weight>0)
                l.weight[pos_idx] += l.weight[pos_idx]*gamma
                return l
        
            else :
                return layer
        
    def relprop(self, activation, layer, R, eps = None, gamma = 0.):
        
        #if not batch : activation = activation.squeeze(0)
        epsilon = 0.
        if eps != None : epsilon = eps
        
        if not activation.requires_grad:
            activation.requires_grad = True
        
        z = epsilon + self.rho(layer, gamma).forward(activation)
        s = R/(z+1e-9)
        (z*s.data).sum().backward()
        c = activation.grad
        ret = activation*c
        return ret
    
    # for deal with input's first layer
    def relprop_ws(self, layer, R):
        w_s = (layer.weight*layer.weight)
        z = w_s.sum(dim = (1, 2, 3))
        s = (R/(z+1e-9))
        R_p = s.matmul(w_s)
        return R_p
    
    # for deal with input's first layer
    def relprop_zb(self, layer, R):
        x = self.model_activation[-1].clone()
        l = torch.full(tuple(x.shape), x.min()).to(device)
        h = torch.full(tuple(x.shape), x.max()).to(device)
        f_p = copy.deepcopy(layer)
        f_n = copy.deepcopy(layer)
        with torch.no_grad():
            f_p.weight[torch.where(f_p.weight<0)] = 0.
            f_n.weight[torch.where(f_n.weight>0)] = 0.
        x.requires_grad = True
        l.requires_grad = True
        h.requires_grad = True
        z = layer.forward(x)-f_p.forward(l)-f_n.forward(h)
        s = R/(z+1e-9)
        (z*s.data).sum().backward()
        return x*x.grad+l*l.grad+h*h.grad
    
    def get_relevance_map(self, image_batch, gamma = 0., epsilon = 0.):
        self.model.eval()
        self.epsilon = epsilon
        self.gamma = gamma
        # when forward pass, each layer's activation is saved
        output = self.forward(image_batch).data
        _, pred = torch.max(output, dim = 1)
        # zero-padding on output value except prediction
        out = torch.full(output.size(), 0.).to(device)
        for i in range(output.size(0)):
            out[i][pred[i]] = output[i][pred[i]]
        activation = self.model_activation[:-1]
        activation.insert(0, out)
        R = out
        flatten_flag = True
        # set flag to flatten when linear-to-conv2d
        for n_layer in range(len(self.model_layers)):
            print(n_layer, len(self.model_activation))
            
            if n_layer < len(self.model_activation)-1:
                act = activation[n_layer+1]
                
            layer = self.model_layers[n_layer]
            
            if n_layer == len(self.model_layers)-1:
                R_score = self.relprop_zb(layer, R)
                print('last', R_score.shape, R_score.sum())
                
            else :
                # Managing special case, when conv-layer to fc-layer
                if isinstance(self.model_layers[n_layer], nn.Linear) and (isinstance(self.model_layers[n_layer+1], nn.Conv2d) or isinstance(self.model_layers[n_layer+1], nn.AvgPool2d)):
                    act = activation[n_layer+1].view(activation[n_layer+1].size(0), -1)
                elif (isinstance(self.model_layers[n_layer], nn.Conv2d) or isinstance(self.model_layers[n_layer], nn.AvgPool2d)) and isinstance(self.model_layers[n_layer-1], nn.Linear):
                    R = R.view(activation[n_layer].size())
                
                # With layer position, applying different relevance propagation rules (Low/Middle/Upper layers)
                if n_layer < 3:
                    R_score = self.relprop(act, layer, R, eps = 0.0)
                elif 3 <= n_layer and n_layer < 11:
                    R_score = self.relprop(act, layer, R, eps = self.epsilon)
                else :
                    R_score = self.relprop(act, layer, R, eps = self.epsilon, gamma = self.gamma)
                print('intermediate', R_score.shape, R_score.sum(), '\n\n')
                
            R = R_score
        return R, pred
            