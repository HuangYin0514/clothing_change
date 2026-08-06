import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecoupleModule(nn.Module):
    """
    通过可学习的 2D 频域掩膜预测器，从幅值谱中学习出过滤服装高频噪声的掩膜，
    保留身份边缘并重构空域净特征。
    """

    def __init__(self, in_channels):
        super().__init__()
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 2D 快速傅里叶变换 (FFT) 正向变换
        fft_x = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

        amplitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        # 从幅值谱学习自适应频域掩膜
        mask = self.mask_generator(amplitude)

        # 只抑制幅值，相位保持不变（关键改进）
        amp_clean = amplitude * mask
        fft_clean = amp_clean * torch.exp(1j * phase)

        # 逆傅里叶变换 (IFFT) 重构空域净特征
        x_clean = torch.fft.irfft2(fft_clean, dim=(-2, -1), norm="ortho")
        return x_clean + x


# =====================================================================
# 6. 代码前向运行与验证测试脚本
# =====================================================================
if __name__ == "__main__":
    inputs = torch.randn(2, 2048, 16, 8)  # 模拟输入特征图
    model = FrequencyDecoupleModule(in_channels=2048)
    outputs = model(inputs)
    print(outputs.shape)  # 输出特征图形状
