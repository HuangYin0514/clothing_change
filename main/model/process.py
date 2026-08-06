import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecoupleModule(nn.Module):
    """
    改进点：
    1. 引入高斯低通先验 (Gaussian Low-Pass Prior)，引导模型明确低频(身份)与高频(服装)的分界；
    2. 使用 1x1 频域通道-幅值注意力 (Frequency-Channel Attention) 替换 3x3 空间卷积，避免频谱尺寸错位；
    3. 增加极小残差缩放因子 (Gamma)，防止训练初期频域滤波大幅破坏特征表征。
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels

        # 频域通道注意力：提取频域下的全局特征上下文
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # 可学习的频域动态掩膜调整器
        self.mask_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.Sigmoid())

        # 可学习可调残差权重，初始接近于 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def _get_gaussian_lowpass_filter(self, shape, device, sigma=0.25):
        """生成标准化的 2D 高斯低通掩膜先验"""
        H, W = shape
        fy = torch.linspace(-1, 1, H, device=device).view(-1, 1).repeat(1, W)
        fx = torch.linspace(0, 1, W, device=device).view(1, -1).repeat(H, 1)  # rfft2 宽度只有一半
        radius_sq = fx**2 + fy**2
        gaussian_mask = torch.exp(-radius_sq / (2 * (sigma**2)))
        return gaussian_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W//2 + 1]

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 2D 实数 FFT
        fft_x = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        amplitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        # 2. 物理先验引导 + 自适应掩膜生成
        gauss_prior = self._get_gaussian_lowpass_filter((H, W // 2 + 1), x.device)
        channel_weights = self.freq_attn(amplitude)
        learned_mask = self.mask_conv(amplitude)

        # 结合固定高斯先验与动态自适应掩膜
        dynamic_mask = learned_mask * channel_weights * gauss_prior

        # 3. 幅值抑制与复数重构
        amp_clean = amplitude * dynamic_mask
        fft_clean = torch.polar(amp_clean, phase)  # 相比 torch.exp(1j*phase) 更稳定且高效

        # 4. 逆傅里叶变换重构
        x_freq_filtered = torch.fft.irfft2(fft_clean, s=(H, W), dim=(-2, -1), norm="ortho")

        # 5. 带有可学习权重的特征融合
        return x + self.gamma * x_freq_filtered


# =====================================================================
# 6. 代码前向运行与验证测试脚本
# =====================================================================
if __name__ == "__main__":
    inputs = torch.randn(2, 2048, 16, 8)  # 模拟输入特征图
    model = FrequencyDecoupleModule(in_channels=2048)
    outputs = model(inputs)
    print(outputs.shape)  # 输出特征图形状
