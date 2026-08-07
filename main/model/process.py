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


# =====================================================================
# 6. 代码前向运行与验证测试脚本
# =====================================================================
if __name__ == "__main__":
    inputs = torch.randn(2, 2048, 16, 8)  # 模拟输入特征图
    model = FrequencyDecoupleModule(in_channels=2048)
    outputs = model(inputs)
    print(outputs.shape)  # 输出特征图形状
