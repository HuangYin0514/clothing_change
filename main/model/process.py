import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecoupleModule(nn.Module):
    """
    通过可学习的 2D 频域掩膜预测器，从幅值谱中学习出过滤服装高频噪声的掩膜，
    保留身份边缘并重构空域净特征。
    """

    def __init__(self, in_channels, shape):
        h, w = shape

        super().__init__()
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.fft = lambda x: torch.fft.fftshift(torch.fft.rfft2(x, norm="ortho"), dim=(-2))
        self.ifft = lambda x: torch.fft.irfft2(torch.fft.ifftshift(x, dim=(-2)), s=(h, w), norm="ortho")

    def forward(self, x):
        # 1. 2D 快速傅里叶变换 (FFT) 正向变换
        fft_x = self.fft(x)
        amplitude = torch.abs(fft_x)

        # 2. 从幅值谱学习自适应频域掩膜
        mask = self.mask_generator(amplitude)

        # 3. 频域掩膜过滤 (剔除高频换衣干扰)
        fft_clean = fft_x * mask

        # 4. 逆傅里叶变换 (IFFT) 重构空域净特征
        x_clean = self.ifft(fft_clean)
        return x_clean + x


# =====================================================================
# 6. 代码前向运行与验证测试脚本
# =====================================================================
if __name__ == "__main__":
    inputs = torch.randn(2, 2048, 16, 8)  # 模拟输入特征图
    model = FrequencyDecoupleModule(in_channels=2048, shape=(16, 8))  # 初始化模块
    outputs = model(inputs)
    print(outputs.shape)  # 输出特征图形状
