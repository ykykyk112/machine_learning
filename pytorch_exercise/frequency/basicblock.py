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
        self.sum_factor = torch.nn.Parameter(torch.zeros((1, self.out_channels, 1, 1)), requires_grad = True)
        #self.sum_factor = torch.nn.Parameter(torch.tensor([0.]), requires_grad = True)
        #self.comp_conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.pooling_kernel_size = 2
                
        self.feed_forward = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding)
        self.second_forward = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding)
        
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


    def forward(self, x, heatmap = None):

        # get grad-cam from network, which has parameters trained previous epoch

        # first conv_block
        ret_first_forward = self.feed_forward(x)
        ret_first_forward = self.first_batch_relu(ret_first_forward)
        ret_pooling = self.first_max_pooling(ret_first_forward)
        
        # get boundary feature map
        ret_upsample = self.up_sampling(ret_pooling)
        
        # second conv_block
        ret_substract = ret_first_forward - ret_upsample

        #with torch.no_grad():
        #   ret_second_forward = self.feed_forward(torch.abs(ret_substract))

        ret_second_forward = self.second_forward(torch.abs(ret_substract))

        ret_second_forward = self.second_batch_relu(ret_second_forward)
        ret_second_forward = self.second_max_pooling(ret_second_forward)
        

        if self.comp_mode == 'C' or self.comp_mode == 'c':
            ret_concat = torch.cat([ret_pooling, ret_second_forward], dim = 1)
            ret_reduction = self.conv_compression(ret_concat)
            return ret_reduction

        elif (self.comp_mode == 'W' or self.comp_mode == 'w') and heatmap != None:

            self.upsample = nn.Upsample(size = ret_second_forward.size(2), mode = 'bilinear', align_corners=False)
            heatmap_upsample = self.upsample(heatmap)
            ret_dot = ret_second_forward * heatmap_upsample
            ret = ret_pooling + self.sum_factor*(ret_dot)
            return ret


        elif self.comp_mode == 'S' or self.comp_mode == 's':
            ret_rescaled = ret_second_forward * (ret_pooling.max()/ret_second_forward.max())
            return ret_pooling + ret_rescaled

        elif (self.comp_mode == 'W' or self.comp_mode == 'w') and heatmap == None:
            #with torch.no_grad():
            c_max, c_min = torch.amax(ret_second_forward, dim = (1, 2, 3)).unsqueeze(1).unsqueeze(1).unsqueeze(1), torch.amin(ret_second_forward, dim = (1, 2, 3)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            ret_rescaled = (ret_second_forward - c_min) / ((c_max - c_min)+1e-15)
            ret = ret_pooling + self.sum_factor*(ret_second_forward)
            return ret

class BoundaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1):
        super(BoundaryConv2d, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = kernel_size
        self.stride, self.padding = stride, padding
        self.pooling_kernel_size = 2
        self.boundary = None
                
        self.feed_forward = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding)
        
        self.batchnorm_relu = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True)
        )
        
        self.max_pooling = nn.MaxPool2d(self.pooling_kernel_size)
        
        self.up_sampling = nn.Sequential(
            nn.Upsample(scale_factor=self.pooling_kernel_size, mode = 'bilinear', align_corners=False),
            nn.ReLU(True)
        )



    def forward(self, x):

        # get grad-cam from network, which has parameters trained previous epoch

        # first conv_block
        ret_first_forward = self.feed_forward(x)
        ret_first_forward = self.batchnorm_relu(ret_first_forward)
        ret_pooling = self.max_pooling(ret_first_forward)
        
        # get substracted
        ret_upsample = self.up_sampling(ret_pooling)
        self.boundary = torch.abs(ret_first_forward - ret_upsample)

        return ret_pooling

class InceptionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionConv2d, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
                
        self.one_by_one = nn.Conv2d(in_channels//4, out_channels/4, 1, 1, 0)
        self.three_by_three = nn.Conv2d(in_channels//4, out_channels/4, 3, 1, 1)
        self.five_by_five = nn.Conv2d(in_channels//4, out_channels/4, 5, 1, 2)
        self.seven_by_seven = nn.Conv2d(in_channels//4, out_channels/4, 7, 1, 3)


    def forward(self, x):

        # get separated channel size
        channel = x.size(1)/4

        # forward in each convolutional layers
        y_0 = self.one_by_one(x[:, :channel*1, :, :])
        y_1 = self.three_by_three(x[:, channel*1:channel*2, :, :])
        y_2 = self.five_by_five(x[:, channel*2:channel*3, :, :])
        y_3 = self.seven_by_seven(x[:, channel*3:channel*4, :, :])
        
        # concatenate each result
        y = torch.cat([y_0, y_1, y_2, y_3], dim = 1)

        return y

        