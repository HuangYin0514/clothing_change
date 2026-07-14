from typing import Optional

import torch
import torch.nn as nn

from .bn_neck import BN_Neck


class Channel_Pool_Attention(nn.Module):
    """
    Pooling based Channel Attention Module
    Args:
        pool_type: pooling mode, support ["avg", "max"]
        reduction: channel reduction ratio
        feature_dim: input feature channel number (MUST be positive integer)
    """

    def __init__(self, pool_type: str = "avg", reduction: int = 16, feature_dim: int = -1):
        super().__init__()
        self.pool_type = pool_type.lower()
        self.feature_dim = feature_dim
        self.reduction = reduction

        # 合法性校验
        if self.pool_type not in ("avg", "max"):
            raise ValueError(f"pool_type only support 'avg' or 'max', got {pool_type}")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be set to positive channel number!")

        # 选择池化层
        if self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)

        # 通道精炼模块
        self.refinement = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // reduction, self.feature_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat shape: [B, C, H, W]
        B, C, H, W = feat.shape
        # 全局池化 + flatten
        pool_feat = self.pool(feat).flatten(1)
        # 生成注意力权重 [B,C,1,1]
        attn_weight = self.refinement(pool_feat).view(B, C, 1, 1)
        # 特征加权
        return feat * attn_weight
