# Hybrid Gradient Enhancement Module 优化版
import torch
import torch.nn as nn
import torch.nn.functional as F


class HGE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        assert kernel_size == 3, "HGE only supports kernel_size=3 for sobel branches"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_log = 2

        self.fc = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # 可训练基础参数 (out_c, in_c, 3, 3)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        # 固定方向模板 register_buffer
        self.register_buffer("template_0", torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32))
        self.register_buffer("template_45", torch.tensor([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=torch.float32))
        self.register_buffer("template_90", torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32))
        self.register_buffer("template_135", torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=torch.float32))

        # LoG mask 预计算，不再forward内求mask
        template_log = torch.zeros((5, 5), dtype=torch.float32)
        template_log[[0, 1, 1, 2, 2, 3, 3, 4], [2, 1, 3, 0, 4, 1, 3, 2]] = 1.0
        template_log[[1, 2, 2, 3], [2, 1, 3, 2]] = 2.0
        template_log[2, 2] = 4.0
        self.register_buffer("template_log", template_log)
        self.register_buffer("mask_a1", (template_log == 4.0))
        self.register_buffer("mask_a2", (template_log == 2.0))
        self.register_buffer("mask_a3", (template_log == 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out_c = self.out_channels

        # 提取缩放系数 [out_c, in_c, 1, 1]
        sob0 = self.weight[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
        sob45 = self.weight[:, :, 0, 1].unsqueeze(-1).unsqueeze(-1)
        sob90 = self.weight[:, :, 0, 2].unsqueeze(-1).unsqueeze(-1)
        sob135 = self.weight[:, :, 1, 0].unsqueeze(-1).unsqueeze(-1)
        gl_coeff = self.weight[:, :, 2, 2].unsqueeze(-1).unsqueeze(-1)

        # ---------------- 4路Sobel 3x3核 ----------------
        w0 = sob0 * self.template_0
        w45 = sob45 * self.template_45
        w90 = sob90 * self.template_90
        w135 = sob135 * self.template_135

        # 4路合并卷积核 [4*out_c, in_c, 3,3]
        w_sobel = torch.cat([w0, w45, w90, w135], dim=0)
        feat_sobel = F.conv2d(x, w_sobel, stride=self.stride, padding=self.padding)
        # feat_sobel: [B,4Oc,H,W] 拆分成4个分支
        x0, x45, x90, x135 = feat_sobel.chunk(4, dim=1)

        # ---------------- LoG 5x5核 ----------------
        a1 = gl_coeff * 4.0
        a2 = -a1 / 8.0
        a3 = -a1 / 16.0
        log_kernel = self.mask_a1 * a1 + self.mask_a2 * a2 + self.mask_a3 * a3
        x_gl = F.conv2d(x, log_kernel, stride=self.stride, padding=self.padding_log)

        concat_feat = torch.cat([x0, x45, x90, x135, x_gl], dim=1)
        out = self.fc(concat_feat)
        return out


if __name__ == "__main__":
    inputs = torch.randn(1, 2048, 16, 8)
    hge = HGE(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
    outputs = hge(inputs)
    print(f"output shape: {outputs.shape}")
