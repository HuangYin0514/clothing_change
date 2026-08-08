import torch
import torch.nn as nn

from .layer import BN_Neck, GeneralizedMeanPoolingP, Linear_Classifier
from .net import resnet50, resnet50_ibn_a
from .process import FrequencyDecoupleModule


# Backbone_R50 ------------------------------
class Backbone_R50(nn.Module):
    def __init__(self, backbone_type):
        super().__init__()

        resnet = None
        if backbone_type == "resnet50":
            resnet = resnet50(pretrained=True)
        if backbone_type == "resnet50_ibn_a":
            resnet = resnet50_ibn_a(pretrained=True)

        # Modifiy backbone
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)

        # Backbone structure
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer1 = resnet.layer1  # 3 blocks / 256, 96, 48
        self.layer2 = resnet.layer2  # 4 blocks / 512, 48, 24
        self.layer3 = resnet.layer3  # 6 blocks / 1024, 24, 12
        self.layer4 = resnet.layer4  # 3 blocks / 2048, 24, 12

        # self.pool_att_0 = Pool_Attention(pool_type="max", feature_dim=2048)
        # self.pool_att_1 = Pool_Attention(pool_type="avg", feature_dim=2048)

        self.fd_l2 = FrequencyDecoupleModule(shape=(512, 48, 24), ratio=0.2)
        self.fd_l3 = FrequencyDecoupleModule(shape=(1024, 24, 12), ratio=0.2)

    def forward(self, img):
        out = self.layer0(img)
        res0_featmap = out

        out = self.layer1[0](out)
        out = self.layer1[1](out)
        out = self.layer1[2](out)
        res1_featmap = out

        out = self.layer2[0](out)
        out = self.layer2[1](out)
        out = self.layer2[2](out)
        out = self.layer2[3](out)
        out = self.fd_l2(out)
        res2_featmap = out

        out = self.layer3[0](out)
        out = self.layer3[1](out)
        out = self.layer3[2](out)
        out = self.layer3[3](out)
        out = self.layer3[4](out)
        out = self.layer3[5](out)
        out = self.fd_l3(out)
        res3_featmap = out

        out = self.layer4[0](out)
        out = self.layer4[1](out)
        out = self.layer4[2](out)
        res4_featmap = out

        return res0_featmap, res1_featmap, res2_featmap, res3_featmap, res4_featmap


class ReID_Net(nn.Module):

    def __init__(self, config, num_pid):
        super().__init__()
        self.config = config

        BACKBONE_TYPE = config.MODEL.BACKBONE_TYPE

        # ------------- Backbone -----------------------
        self.backbone = Backbone_R50(BACKBONE_TYPE)

        # ------------- Global -----------------------
        self.GLOBAL_DIM = 2048
        self.global_pool = GeneralizedMeanPoolingP()
        self.global_bn_neck = BN_Neck(self.GLOBAL_DIM)
        self.global_classifier = Linear_Classifier(self.GLOBAL_DIM, num_pid)

    def heatmap(self, img):
        B, C, H, W = img.shape
        res0_featmap, res1_featmap, res2_featmap, res3_featmap, res4_featmap = self.backbone(img)
        return res4_featmap

    def forward(self, img):
        B, C, H, W = img.shape

        # ------------- Global -----------------------
        res0_featmap, res1_featmap, res2_featmap, res3_featmap, res4_featmap = self.backbone(img)

        if not self.training:
            eval_feat_meter = []

            # ------------- Global -----------------------
            global_feat = self.global_pool(res4_featmap).view(B, self.GLOBAL_DIM)
            global_bn_feat = self.global_bn_neck(global_feat)
            eval_feat_meter.append(global_bn_feat)

            eval_feat = torch.cat(eval_feat_meter, dim=1)
            return eval_feat

        return res0_featmap, res1_featmap, res2_featmap, res3_featmap, res4_featmap
