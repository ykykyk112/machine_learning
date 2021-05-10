import torch
import torch.nn as nn

'''
Class Implementation :

    * Grad-CAM class, which is applicable any network.
    * Forward_hook & backward_hook register feature map and gradient when data is passed.
    * You should check index of weight before register forward hook and backward hook and then modify index.

Function Implemantation :
    
    def get_cam :
        * Input image and label should be shape of batch.
        * Using conv-layer's feature map and gradient, get heat-map.
        * Return heat-map size is match with input size.
        * Return heat-map's pixel get more higher value when pixel has more importance for prediction input's class .
    
    def forward_hook :
        * Function get registered conv-layer's feature map, when data passes that layer in forwarding.
        * Function get registered conv-layer's gradient of predicted class value, when gradient passes that layer in backpropagation.
'''

class grad_cam() :
    def __init__(self, model):
        # hook on last conv-layer (model.features[40])
        self.model = model
        self.hook_history = []
        for m in reversed(list(self.model.modules())):
            if isinstance(m, nn.Conv2d):
                self.hook_history.append(m.register_forward_hook(self.forward_hook))
                self.hook_history.append(m.register_full_backward_hook(self.backward_hook))
                break

    def remove_hook(self):
        for h in self.hook_history:
            h.remove()
        self.hook_history = []
        
    # forward_result는 forward시 해당 conv_layer의 출력 결과인 feature-map이다.
    def forward_hook(self, _, input_image, output):
        self.forward_result = torch.squeeze(output)
    # backward_result는 backpropagation에서 구한 y_c를 feature-map의 각 elem으로 편미분한 gradient이다.
    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])
        
    def get_cam(self, image_batch, label_batch):
        # heatmap을 저장할 empty tensor
        ret = torch.empty((image_batch.size(0), 14, 14))
        ret_pred = torch.empty((image_batch.size(0)))
        self.model.eval()
        for idx, (image, label) in enumerate(zip(image_batch, label_batch)):
            # forward의 input은 batch 형태여야 하므로, batch_dim을 추가해주고, output의 batch_dim은 제거해준다.
            pred = self.model.forward(image.unsqueeze(0)).squeeze()
            _, pred_max = torch.max(pred, dim = 0)
            pred[int(pred_max)].backward()
            # predict image에 대한 forward_result, backward_result 생성 완료
            a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)
            # a_k-(512, 1, 1) * feature_map-(512, 7, 7) 이후 sum을 통해 S_k=(7, 7)로 만들어준다.
            cam = torch.sum(a_k * torch.nn.functional.relu(self.forward_result), dim=0)
            cam_relu = torch.nn.functional.relu(cam)
            ret_pred[idx] = pred_max
            ret[idx] = cam_relu
        # upsampling input은 4-dimension이다.
        upsampling = nn.Upsample(size = image_batch.size(2), mode = 'bilinear', align_corners=False)
        ret_upsampled = upsampling(ret.unsqueeze(0)).squeeze()
        return ret_upsampled, ret_pred

    def get_label_cam(self, image_batch, label_batch):
        # heatmap을 저장할 empty tensor
        ret = torch.empty((image_batch.size(0), 14, 14))
        ret_pred = torch.empty((image_batch.size(0)))
        self.model.eval()
        for idx, (image, label) in enumerate(zip(image_batch, label_batch)):
            # forward의 input은 batch 형태여야 하므로, batch_dim을 추가해주고, output의 batch_dim은 제거해준다.
            pred = self.model.forward(image.unsqueeze(0)).squeeze()
            pred[int(label)].backward()
            # predict image에 대한 forward_result, backward_result 생성 완료
            a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)
            # a_k-(512, 1, 1) * feature_map-(512, 7, 7) 이후 sum을 통해 S_k=(7, 7)로 만들어준다.
            cam = torch.sum(a_k * torch.nn.functional.relu(self.forward_result), dim=0)
            cam_relu = torch.nn.functional.relu(cam)
            ret_pred[idx] = label
            ret[idx] = cam_relu
        # upsampling input은 4-dimension이다.
        upsampling = nn.Upsample(size = image_batch.size(2), mode = 'bilinear', align_corners=False)
        ret_upsampled = upsampling(ret.unsqueeze(0)).squeeze()
        return ret_upsampled, ret_pred