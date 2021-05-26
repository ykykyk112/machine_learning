from torch import nn
from pytorch_exercise.frequency.basicblock import RecoverConv2d

class MySequential(nn.Sequential):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], nn.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward_cam(self, x, heatmap):
        for module in self:
            if isinstance(module, RecoverConv2d):
                x = module.forward_dot(x, heatmap)
            else:
                x = module(x)
        return x

    def forward(self, x):
        for module in self:
            x = module(x)
        return x

