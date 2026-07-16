import torch
import torch.nn as nn

from .backbone import Backbone_R50
from .layer import BN_Neck, GeneralizedMeanPoolingP, Linear_Classifier


class ReID_Net(nn.Module):

    def __init__(self, config, num_pid):
        super().__init__()
        self.config = config

        BACKBONE_TYPE = config.MODEL.BACKBONE_TYPE

        # ------------- Backbone -----------------------
        self.backbone = Backbone_R50(BACKBONE_TYPE)

        # ------------- Global -----------------------
        self.GLOBAL_DIM = 2048
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_bn_neck = BN_Neck(self.GLOBAL_DIM)
        self.global_classifier = Linear_Classifier(self.GLOBAL_DIM, num_pid)

    def heatmap(self, img):
        B, C, H, W = img.shape
        backbone_feat_map = self.backbone(img)
        return backbone_feat_map

    def forward(self, img):
        B, C, H, W = img.shape

        # ------------- Global -----------------------
        backbone_feat_map = self.backbone(img)
        global_feat = self.global_pool(backbone_feat_map).view(B, self.GLOBAL_DIM)
        global_bn_feat = self.global_bn_neck(global_feat)

        if self.training:
            return backbone_feat_map, global_feat, global_bn_feat
        else:
            eval_feat_meter = []
            # ------------- Global -----------------------
            eval_feat_meter.append(global_bn_feat)
            eval_feat = torch.cat(eval_feat_meter, dim=1)
            return eval_feat
