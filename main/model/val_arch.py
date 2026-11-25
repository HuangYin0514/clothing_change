import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTForward(nn.Module):
    """
    2D Haar DWT 前向分解 J=1
    输入: x [B,C,H,W] H,W为偶数
    返回: LL: [B,C,H//2,W//2], high: (HL, LH, HH)
    """

    def __init__(self):
        super().__init__()
        # haar 滤波器
        self.register_buffer("h0", torch.tensor([1.0, 1.0]) / 2)
        self.register_buffer("h1", torch.tensor([1.0, -1.0]) / 2)

    def forward(self, x):
        B, C, H, W = x.shape
        # 行滤波 + 下采样
        x = x.reshape(B * C, 1, H, W)
        # 水平方向
        ll_h = F.conv2d(x, self.h0.view(1, 1, 1, 2), stride=(1, 2), padding=0)
        lh_h = F.conv2d(x, self.h1.view(1, 1, 1, 2), stride=(1, 2), padding=0)
        # 垂直方向
        ll = F.conv2d(ll_h, self.h0.view(1, 1, 2, 1), stride=(2, 1), padding=0)
        hl = F.conv2d(ll_h, self.h1.view(1, 1, 2, 1), stride=(2, 1), padding=0)
        lh = F.conv2d(lh_h, self.h0.view(1, 1, 2, 1), stride=(2, 1), padding=0)
        hh = F.conv2d(lh_h, self.h1.view(1, 1, 2, 1), stride=(2, 1), padding=0)

        ll = ll.reshape(B, C, H // 2, W // 2)
        hl = hl.reshape(B, C, H // 2, W // 2)
        lh = lh.reshape(B, C, H // 2, W // 2)
        hh = hh.reshape(B, C, H // 2, W // 2)
        return ll, (hl, lh, hh)


class DWTInverse(nn.Module):
    """
    2D Haar IDWT 逆重建，对应上面DWTForward
    输入：LL, (HL, LH, HH)
    输出：重建x [B,C,H,W]
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("h0", torch.tensor([1.0, 1.0]))
        self.register_buffer("h1", torch.tensor([1.0, -1.0]))

    def forward(self, dwt_out):
        ll, (hl, lh, hh) = dwt_out
        B, C, h2, w2 = ll.shape
        H, W = h2 * 2, w2 * 2

        ll = ll.reshape(B * C, 1, h2, w2)
        hl = hl.reshape(B * C, 1, h2, w2)
        lh = lh.reshape(B * C, 1, h2, w2)
        hh = hh.reshape(B * C, 1, h2, w2)

        # 垂直上采样+滤波
        ll_up = F.conv_transpose2d(ll, self.h0.view(1, 1, 2, 1), stride=(2, 1), padding=0)
        hl_up = F.conv_transpose2d(hl, self.h1.view(1, 1, 2, 1), stride=(2, 1), padding=0)
        lh_up = F.conv_transpose2d(lh, self.h0.view(1, 1, 2, 1), stride=(2, 1), padding=0)
        hh_up = F.conv_transpose2d(hh, self.h1.view(1, 1, 2, 1), stride=(2, 1), padding=0)

        # 水平上采样+滤波
        ll_final = F.conv_transpose2d(ll_up, self.h0.view(1, 1, 1, 2), stride=(1, 2), padding=0)
        hl_final = F.conv_transpose2d(hl_up, self.h0.view(1, 1, 1, 2), stride=(1, 2), padding=0)
        lh_final = F.conv_transpose2d(lh_up, self.h1.view(1, 1, 1, 2), stride=(1, 2), padding=0)
        hh_final = F.conv_transpose2d(hh_up, self.h1.view(1, 1, 1, 2), stride=(1, 2), padding=0)

        x_rec = ll_final + hl_final + lh_final + hh_final
        x_rec = x_rec.reshape(B, C, H, W)
        return x_rec


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class DualDWTUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualDWTUnit, self).__init__()
        # 小波正变换、逆变换
        self.dwt = DWTForward(J=1, wave="haar", mode="zero")
        self.idwt = DWTInverse(wave="haar", mode="zero")

        self.res_conv = weight_norm(nn.Conv2d(in_channels, out_channels // 2, 1))
        # 低频FFT分支
        self.fft_amp_conv = nn.Sequential(weight_norm(nn.Conv2d(in_channels, out_channels // 2, 1)), nn.LeakyReLU(0.05, inplace=True))
        self.fft_phase_conv = nn.Sequential(weight_norm(nn.Conv2d(in_channels, out_channels // 2, 1)), nn.LeakyReLU(0.05, inplace=True))
        # 高频FFT分支（修复原来的in_channel_33笔误）
        self.high_fft_amp_conv = nn.Sequential(weight_norm(nn.Conv2d(in_channels, out_channels // 2, 1)), nn.LeakyReLU(0.05, inplace=True))
        self.high_fft_phase_conv = nn.Sequential(weight_norm(nn.Conv2d(in_channels, out_channels // 2, 1)), nn.LeakyReLU(0.05, inplace=True))
        self.fuse_high = nn.Sequential(weight_norm(nn.Conv2d(out_channels // 2, out_channels // 2, 1)), nn.LeakyReLU(0.05, inplace=True))
        self.fuse_low = nn.Sequential(weight_norm(nn.Conv2d(in_channels, out_channels // 2, 1)), nn.LeakyReLU(0.05, inplace=True))

    def fft_process(self, x):
        x = x.to(torch.float32)
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) if H % 2 != 0 else 0
        pad_w = (2 - W % 2) if W % 2 != 0 else 0
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        fft = torch.fft.rfft2(x_padded, norm="backward")
        amp = torch.abs(fft) + 1e-8
        phase = torch.angle(fft)

        amp_feat = self.fft_amp_conv(amp)
        phase_feat = self.fft_phase_conv(phase)
        # ✅ 修复复数重构：增加虚部1j
        fused = amp_feat * torch.cos(phase_feat) + 1j * amp_feat * torch.sin(phase_feat)
        res = torch.fft.irfft2(fused, s=x_padded.shape[-2:], norm="backward")
        return res[:, :, :H, :W]

    def high_fft_process(self, x):
        B, C, H, W = x.shape
        newH = H if H % 2 == 0 else H + 1
        newW = W if W % 2 == 0 else W + 1
        x_pad = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)

        fft = torch.fft.rfft2(x_pad, norm="backward")
        amp = torch.abs(fft) + 1e-8
        phase = torch.angle(fft)

        amp_feat = self.high_fft_amp_conv(amp)
        phase_feat = self.high_fft_phase_conv(phase)
        fused = amp_feat * torch.cos(phase_feat) + 1j * amp_feat * torch.sin(phase_feat)
        res = torch.fft.irfft2(fused, s=x_pad.shape[-2:], norm="backward")
        return res[:, :, :H, :W]

    def forward(self, x):
        # DWT分解
        ll, h = self.dwt(x)
        # h[0] shape: [B, C, 3, H/2, W/2]
        hl, lh, hh = h[0].chunk(3, dim=2)
        hl = hl.squeeze(2)
        lh = lh.squeeze(2)
        hh = hh.squeeze(2)

        # 低频分支
        ll_up = F.interpolate(ll, scale_factor=2, mode="bilinear", align_corners=False)
        ll_fft = self.fft_process(ll_up)
        ll_processed = self.res_conv(ll_up) + ll_fft

        # 三路高频各自FFT处理，不提前concat（方案B核心，支持IDWT）
        hl_proc = self.high_fft_process(hl)
        lh_proc = self.high_fft_process(lh)
        hh_proc = self.high_fft_process(hh)
        highs_proc = (hl_proc, lh_proc, hh_proc)

        # pytorch_wavelets IDWT输入格式：(ll, [ (hl,lh,hh) ])
        recon_feat = self.idwt((ll, [highs_proc]))
        return recon_feat


if __name__ == "__main__":
    # ---------------------- 测试样例 ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # 超参
    B = 2
    in_ch = 3
    out_ch = 64
    H, W = 64, 64

    # 构造模块
    model = DualDWTUnit(in_channels=in_ch, out_channels=out_ch).to(device)
    # 构造随机输入
    x = torch.randn(B, in_ch, H, W, device=device, requires_grad=True)

    # 前向推理
    out = model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")

    # 梯度校验（保证可训练，无异常autograd报错）
    loss = out.sum()
    loss.backward()
    print("✅ 梯度反向传播正常，模块可训练")
    print("-" * 50)

    # 再测奇数尺寸（测试padding/插值鲁棒性）
    H2, W2 = 65, 65
    x2 = torch.randn(1, in_ch, H2, W2, device=device)
    out2 = model(x2)
    print(f"Odd size input ({H2},{W2}) -> output shape: {out2.shape}")
    print("✅ 奇数尺寸测试通过")


# if __name__ == "__main__":
#     dwt = DWTForward()
#     idwt = DWTInverse()
#     x = torch.randn(1, 3, 32, 32)
#     ll, highs = dwt(x)
#     x_rec = idwt((ll, highs))
#     print(x.shape, x_rec.shape)
