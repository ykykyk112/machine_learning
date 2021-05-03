import torch
import torch.nn as nn


class RecoverConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, comp_mode = 'W', upsample_mode = True):
        super(RecoverConv2d, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = kernel_size
        self.stride, self.padding = stride, padding
        self.comp_mode = comp_mode
        self.upsample_mode = upsample_mode
        self.sum_factor = torch.nn.Parameter(torch.tensor([0.5]), requires_grad = True)

        self.pooling_kernel_size = 2
                
        self.feed_forward = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding)
        
        self.first_batch_relu = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True)
        )
        
        self.second_batch_relu = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True)
        )
        
        self.first_max_pooling = nn.MaxPool2d(self.pooling_kernel_size)
        self.second_max_pooling = nn.MaxPool2d(self.pooling_kernel_size)
        
        if self.upsample_mode:
            self.up_sampling = nn.Sequential(
                nn.Upsample(scale_factor=self.pooling_kernel_size, mode = 'bilinear', align_corners=False),
                nn.ReLU(True)
            )
        else :
            self.up_sampling = nn.Sequential(
                nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=4, stride = 2, padding = 1),
                nn.ReLU(True)
            )
        
        if self.comp_mode == 'C' or self.comp_mode == 'c':
            self.conv_compression = nn.Sequential(
                nn.Conv2d(2 * self.out_channels, self.out_channels, kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(True)
            )
        
        
    def forward(self, x):
        # first conv_block
        ret_first_forward = self.feed_forward(x)
        ret_first_forward = self.first_batch_relu(ret_first_forward)
        ret_pooling = self.first_max_pooling(ret_first_forward)
        
        # get boundary feature map
        ret_upsample = self.up_sampling(ret_pooling)
        
        # second conv_block
        with torch.no_grad():
            ret_substract = ret_first_forward - ret_upsample
            ret_second_forward = self.feed_forward(torch.abs(ret_substract))
        ret_second_forward = self.second_batch_relu(ret_second_forward)
        ret_second_forward = self.second_max_pooling(ret_second_forward)
        

        if self.comp_mode == 'C' or self.comp_mode == 'c':
            ret_concat = torch.cat([ret_pooling, ret_second_forward], dim = 1)
            ret_reduction = self.conv_compression(ret_concat)
            return ret_reduction

        elif self.comp_mode == 'W' or self.comp_mode == 'w':
            with torch.no_grad():
                ret_rescaled = ret_second_forward * (ret_pooling.max()/ret_second_forward.max())
            #print('sum_factor : ', self.sum_factor.item())
            return ret_pooling + self.sum_factor*(ret_rescaled)
            #return ret_pooling + 0.1*(ret_rescaled)


        elif self.comp_mode == 'S' or self.comp_mode == 's':
            ret_rescaled = ret_second_forward * (ret_pooling.max()/ret_second_forward.max())
            return ret_pooling + ret_rescaled
