import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecoupleModule(nn.Module):

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels

        # 提取频域下的全局特征上下文
        self.freq_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # 可学习可调残差权重，初始接近于 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 2D 实数 FFT
        fft_x = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        amplitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        # 2. 自适应掩膜生成
        dynamic_mask = self.freq_attn(amplitude)

        # 3. 幅值抑制与复数重构
        amp_clean = amplitude * dynamic_mask
        fft_clean = torch.polar(amp_clean, phase)

        # 4. 逆傅里叶变换重构
        x_freq_filtered = torch.fft.irfft2(fft_clean, s=(H, W), dim=(-2, -1), norm="ortho")

        # 5. 带有可学习权重的特征融合
        return x + self.gamma * x_freq_filtered


class DWT2D(nn.Module):
    """原生 PyTorch 实现的 2D Haar 离散小波变换"""

    def __init__(self):
        super(DWT2D, self).__init__()

    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) / 2.0  # 低频近似 (结构)
        LH = (x00 - x01 + x10 - x11) / 2.0  # 水平高频 (边缘)
        HL = (x00 + x01 - x10 - x11) / 2.0  # 垂直高频 (边缘)
        HH = (x00 - x01 - x10 + x11) / 2.0  # 对角高频 (细节)
        return LL, LH, HL, HH


class SpatialFrequencyLocalAlignment(nn.Module):
    """
    Spatial-Frequency Local Alignment via 2D-DWT (SFLA)
    对应痛点二：解决传统全局频域变换丢失空间位置信息的问题
    """

    def __init__(self, in_channels):
        super(SpatialFrequencyLocalAlignment, self).__init__()
        self.dwt = DWT2D()

        # 融合小波 4 个子带的卷积映射 (LL, LH, HL, HH 拼接后维度为 4*C)
        self.subband_conv = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)

        # 空-频交叉注意力组件
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_spatial):
        B, C, H, W = x_spatial.shape

        # 1. 2D 小波分解，获得兼具空-频局域性的子带
        LL, LH, HL, HH = self.dwt(x_spatial)
        wavelet_subbands = torch.cat([LL, LH, HL, HH], dim=1)  # [B, 4C, H/2, W/2]
        freq_feat = self.subband_conv(wavelet_subbands)  # [B, C, H/2, W/2]

        # 2. 将空间特征下采样对齐尺寸
        spatial_down = F.interpolate(x_spatial, size=(H // 2, W // 2), mode="nearest")

        # 3. 空间-频域交叉注意力计算
        proj_query = self.query_conv(spatial_down).view(B, -1, (H // 2) * (W // 2)).permute(0, 2, 1)  # [B, N, C']
        proj_key = self.key_conv(freq_feat).view(B, -1, (H // 2) * (W // 2))  # [B, C', N]
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N]
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(freq_feat).view(B, -1, (H // 2) * (W // 2))  # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H // 2, W // 2)

        # 4. 残差连接与尺寸还原
        aligned_feat = x_spatial + self.gamma * F.interpolate(out, size=(H, W), mode="nearest")
        return aligned_feat


# =====================================================================
# 2. 幅度-相位结构对齐与增强模块 (Phase-Consistent Alignment, PCSA)
# 针对难题 3：保留相位（几何结构/身份），混洗/增强幅度（衣物纹理/风格）
# =====================================================================
class PhaseConsistent_Alignment(nn.Module):
    def __init__(self):
        super(PhaseConsistent_Alignment, self).__init__()

    def forward(self, x, amp, phase):
        """
        x: [B, C, H, W]
        amp, phase: 来自 FFT 的幅度谱和相位谱
        """
        B, C, H, W = x.size()

        if self.training and B > 1:
            # 幅度混洗增强 (Amplitude Swapping)
            # 随机将 Batch 内不同行人的幅度谱交换，但保留各自的相位谱
            rand_idx = torch.randperm(B, device=x.device)
            amp_swapped = amp[rand_idx]

            # 使用交换后的幅度与原相位合成新的复数频谱: A * exp(i * P)
            fft_swapped = torch.polar(amp_swapped, phase)
            x_amp_swapped = torch.fft.irfft2(fft_swapped, s=(H, W), norm="ortho")
        else:
            x_amp_swapped = x

        return x_amp_swapped


if __name__ == "__main__":
    inputs = torch.randn(2, 2048, 16, 8)  # 模拟输入特征图
    model = FrequencyDecoupleModule(in_channels=2048)
    outputs = model(inputs)
    print(outputs.shape)  # 输出特征图形状

    inputs = torch.randn(2, 2048, 16, 8)  # 模拟输入特征图
    model = SpatialFrequencyLocalAlignment(in_channels=2048)
    outputs = model(inputs)
    print(outputs.shape)  # 输出特征图形状

    def set_seed(seed=42):
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def test_reproduce():
        set_seed(42)
        module = SpatialFrequencyLocalAlignment(in_channels=16)
        x = torch.randn(2, 16, 64, 64)
        out1 = module(x)

        set_seed(42)
        module2 = SpatialFrequencyLocalAlignment(in_channels=16)
        module2.load_state_dict(module.state_dict())
        out2 = module2(x)

        print("max diff:", torch.max(torch.abs(out1 - out2)).item())

    test_reproduce()
