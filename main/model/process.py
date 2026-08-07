import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_masks(h, w, N=None, linspaces=None, mode="square"):
    assert mode in ["square", "rhombus", "circle"], f"mode should be'square', 'rhombus', or 'circle', but got {mode}"
    assert not (N is None and linspaces is None), "either N or linspaces should be provided"

    h_freqs = torch.fft.fftfreq(h)
    h_freqs = torch.fft.fftshift(h_freqs)
    w_freqs = torch.fft.rfftfreq(w)
    hw_freqs = torch.meshgrid(h_freqs, w_freqs, indexing="ij")

    if linspaces is None:
        rs = torch.linspace(0, 1, N + 1)[1:-1].tolist()
    else:
        if isinstance(linspaces, (int, float)):
            rs = [linspaces]
        else:
            rs = linspaces

    if mode == "square":
        masks = []
        for i in range(len(rs)):
            vi = 0.5 * rs[i]

            mask = torch.zeros_like(hw_freqs[0])
            flag = (hw_freqs[0].abs() <= vi) & (hw_freqs[1].abs() <= vi)

            mask[flag] = 1.0
            if i:
                for j in range(i):
                    mask = mask - masks[j]
            masks.append(mask)

        mask = torch.ones_like(hw_freqs[0]) - torch.stack(masks, dim=0).sum(0)
        masks.append(mask)
    elif mode == "rhombus":
        masks = []
        for i in range(len(rs)):
            vi = 0.5 * rs[i]

            mask = torch.zeros_like(hw_freqs[0])
            flag = (
                ((hw_freqs[0].abs() + hw_freqs[1].abs()) <= vi) & ((hw_freqs[0].abs() + hw_freqs[1].abs()) > vi_1)
                if i
                else ((hw_freqs[0].abs() + hw_freqs[1].abs()) <= vi)
            )
            mask[flag] = 1.0

            masks.append(mask)

            vi_1 = vi

        mask = torch.ones_like(hw_freqs[0]) - torch.stack(masks, dim=0).sum(0)
        masks.append(mask)

    else:
        masks = []
        for i in range(len(rs)):
            vi = 0.5 * rs[i]

            mask = torch.zeros_like(hw_freqs[0])
            flag = (
                ((hw_freqs[0] ** 2 + hw_freqs[1] ** 2) <= vi**2) & ((hw_freqs[0] ** 2 + hw_freqs[1] ** 2) > vi_1**2)
                if i
                else ((hw_freqs[0] ** 2 + hw_freqs[1] ** 2) <= vi**2)
            )
            mask[flag] = 1.0

            masks.append(mask)

            vi_1 = vi

        mask = torch.ones_like(hw_freqs[0]) - torch.stack(masks, dim=0).sum(0)
        masks.append(mask)

    return torch.stack(masks, dim=0)  # [N, H, W]


class OSBlock(nn.Module):
    def __init__(self, shape, num_scales=4):
        super().__init__()
        in_channels = shape[0]

        self.num_scales = num_scales
        mid_channels = in_channels // num_scales

        lightconv3x3 = lambda channels: nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))

        self.conv2 = nn.ModuleList([nn.Sequential(*[lightconv3x3(mid_channels) for _ in range(i + 1)]) for i in range(num_scales)])

        self.conv3 = nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=False)

        self.share_attn = nn.Sequential(nn.Conv1d(2, 1, 5, 1, 2), nn.Sigmoid())
        nn.init.zeros_(self.share_attn[-2].bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = 0.0
        for i in range(self.num_scales):
            x2_ = self.conv2[i](x1)
            w = self.share_attn(torch.cat([x2_.mean((2, 3)).view(x.size(0), 1, -1), x2_.amax((2, 3)).view(x.size(0), 1, -1)], dim=1)).view(x.size(0), -1, 1, 1)
            x2 = x2 + x2_ * w

        x3 = self.conv3(x2)
        return x3


class FrequencyDecoupleModule(nn.Module):
    def __init__(self, shape, ratio=0.2):
        super(FrequencyDecoupleModule, self).__init__()
        self.shape = shape
        self.ratio = ratio

        in_dims, h, w = shape

        self.bn = nn.BatchNorm2d(in_dims)
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

        self.hig_os = OSBlock(shape, 4)

        self.conv2 = nn.Sequential(nn.Conv2d(in_dims, 2, 3, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.conv2[-2].bias)

        self.fft = lambda x: torch.fft.fftshift(torch.fft.rfft2(x, norm="ortho"), dim=(-2))
        self.ifft = lambda x: torch.fft.irfft2(torch.fft.ifftshift(x, dim=(-2)), s=(h, w), norm="ortho")

        mask = _get_masks(h, w, linspaces=ratio, mode="square")[0]
        self.h_start = int(torch.argmax(mask, dim=0)[0].item())
        self.h_crop = int(mask.sum(0)[0].item())
        self.w_crop = int(mask.sum(1).max().item())
        self.register_buffer("low_mask", mask)

        self.low_weights = nn.Parameter(torch.randn(in_dims, self.h_crop, self.w_crop, 2) * 0.02)

    def forward(self, x):
        x = x.to(torch.float32)
        x_fft = self.fft(x)
        x_low = (x_fft * self.low_mask).clone()
        x_low[..., self.h_start : self.h_start + self.h_crop, : self.w_crop] = x_fft[
            ..., self.h_start : self.h_start + self.h_crop, : self.w_crop
        ] * torch.view_as_complex(self.low_weights)

        x_hig = x_fft * (1.0 - self.low_mask)
        x_ = torch.cat([x_low, x_hig], dim=0)
        x_ = self.ifft(x_)

        x_low, x_hig = x_.chunk(2, dim=0)

        x_hig = self.hig_os(x_hig)

        sp_attn = self.conv2(x)
        x_ = x_low * sp_attn[:, :1] + x_hig * sp_attn[:, 1:]

        return x + self.bn(x_)


# =====================================================================
# 6. 代码前向运行与验证测试脚本
# =====================================================================
if __name__ == "__main__":
    inputs = torch.randn(2, 2048, 16, 8)  # 模拟输入特征图
    model = FrequencyDecoupleModule(shape=(2048, 16, 8))
    outputs = model(inputs)
    print(outputs.shape)  # 输出特征图形状
