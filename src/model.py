import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM

import segmentation_models_pytorch as smp

# class Detector(nn.Module):

#     def __init__(self):
#         super(Detector, self).__init__()
#         self.net = EfficientNet.from_pretrained(
#             "efficientnet-b4", weights_path="weights/adv-efficientnet-b4-44fb3a87.pth", advprop=True, num_classes=2)
#         self.cel = nn.CrossEntropyLoss()
#         self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=0.001, momentum=0.9)

#     def forward(self, x):
#         x = self.net(x)
#         return x

#     def training_step(self, x, target):
#         for i in range(2):
#             pred_cls = self(x)
#             if i == 0:
#                 pred_first = pred_cls
#             loss_cls = self.cel(pred_cls, target)
#             loss = loss_cls
#             self.optimizer.zero_grad()
#             loss.backward()
#             if i == 0:
#                 self.optimizer.first_step(zero_grad=True)
#             else:
#                 self.optimizer.second_step(zero_grad=True)

#         return pred_first



class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        
        aux_params=dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=1,                 # define number of output labels
        )

        self.net = smp.DeepLabV3Plus(encoder_name="efficientnet-b4", encoder_weights="imagenet", activation='sigmoid', 
                                     in_channels=3, classes=1, aux_params=aux_params)
        # self.net = smp.DeepLabV3Plus(encoder_name="efficientnet-b4", encoder_weights="imagenet", in_channels=3, classes=1)

        # self.net = EfficientNet.from_pretrained(
        #     "efficientnet-b4", weights_path="weights/adv-efficientnet-b4-44fb3a87.pth", advprop=True, num_classes=2)
        # self.loss = None
        # self.cel = nn.CrossEntropyLoss()
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=0.001, momentum=0.9)

    def forward(self, x):
        """
        output_mask: in shape (B, C, W, H)
        output_class, in shape (B, )
        """
        # x = self.net(x)
        # output_mask, output_target = self.net(x)
        output_mask, output_class = self.net(x)

        # print("output_class:", output_class)

        return output_mask, output_class.squeeze()

    def training_step(self, x, target_mask, target_class):
        for i in range(2):
            output_mask, output_class = self(x)
            output_class = output_class
            if i == 0:
                mask_first = output_mask
                class_first = output_class
            # loss_mask = nn.BCELoss()(output_mask, target_mask)
            # loss_class = nn.CrossEntropyLoss()(output_class, target_class)
            
            # loss_mask = nn.BCELoss()(output_mask, target_mask)
            # loss_class = nn.BCELoss()(output_class, target_class)
            # loss = loss_mask * 100 + loss_class  

            loss = Detector.compute_loss(output_mask, target_mask, output_class, target_class)

            self.optimizer.zero_grad()
            loss.backward()
            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)

        return mask_first, class_first
    
    @staticmethod
    def compute_loss(output_mask, target_mask, output_class, target_class):

        # loss_mask = nn.BCELoss()(output_mask, target_mask)
        # loss_class = nn.BCELoss()(output_class, target_class)

        # return loss_mask * 100 + loss_class 

        return nn.BCELoss()(output_class, target_class)