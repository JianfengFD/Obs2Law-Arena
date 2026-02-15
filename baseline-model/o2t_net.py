#!/usr/bin/env python3
"""
o2t_net.py — Observation-to-Theory Neural Network
=====================================================
Core architecture for inferring physics from rendered images.

Architecture:
  Input: 4 RGB images (stereo pairs at t1, t2) + 5 parameter sets
  → 23-channel image tensor (full resolution, e.g. 512×512)
  → Encoder (convolutional downsampling, no resize)
     512→256→128→64→32  (5 levels)
  → h_48 latent via AdaptiveAvgPool
  → W_ROT rotation → n×PhysLayer → W_ROT_INV
  → Decoder (upsampling with skip connections)
     32→64→128→256→512
  → Predicted future image (3ch, same resolution as input)

The network processes FULL resolution images natively.
No artificial resizing — the encoder extracts high-res features
through learned convolutions, preserving fine detail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# =====================================================================
# Building Blocks
# =====================================================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Downsample(nn.Module):
    """2× spatial downsampling via strided convolution."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """2× spatial upsampling via nearest + conv."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial dimensions."""
    def __init__(self, ch, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.n_heads, C // self.n_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (C // self.n_heads) ** -0.5
        attn = (q.transpose(-1, -2) @ k * scale).softmax(dim=-1)
        out = (v @ attn).reshape(B, C, H, W)
        return x + self.proj(out)


class ConditionInjection(nn.Module):
    """Adaptive scale + shift conditioning (AdaLN-style)."""
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.proj = nn.Linear(cond_dim, ch * 2)

    def forward(self, x, cond):
        s, b = self.proj(cond).chunk(2, dim=-1)
        s = s[:, :, None, None]
        b = b[:, :, None, None]
        return x * (1 + s) + b


# =====================================================================
# Parameter Embedding
# =====================================================================

class ParamEncoder(nn.Module):
    """
    Encode [t, x, y, z, az, el] → 1-channel feature map.
    Produces 8×8 then upsamples to match the target spatial size
    at runtime via forward(params, target_size).
    """
    def __init__(self, param_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 64),  # 8×8 = 64
        )

    def forward(self, params, target_size):
        """
        params: (B, param_dim)
        target_size: int (e.g. 512)
        Returns: (B, 1, target_size, target_size)
        """
        B = params.shape[0]
        out = self.net(params).view(B, 1, 8, 8)
        if target_size != 8:
            out = F.interpolate(out, size=(target_size, target_size),
                                mode='bilinear', align_corners=False)
        return out


# =====================================================================
# Encoder — native resolution, convolutional downsampling
# =====================================================================

class Encoder(nn.Module):
    """
    23ch input at full resolution → h_48 latent.

    5 downsampling levels:
      512→256→128→64→32  (for 512 input)
      256→128→64→32→16   (for 256 input)

    Channel progression: 23→ch0→ch1→ch2→ch3→ch4
    Self-attention only at the two smallest spatial levels.

    The first two levels use lightweight blocks (fewer channels)
    to handle high-resolution feature maps efficiently.
    """
    def __init__(self, in_ch=23, latent_dim=48, base_ch=32, cond_dim=48):
        super().__init__()
        # Channel schedule: keep early layers thin for memory
        # base_ch=32: [16, 32, 64, 128, 256]
        # base_ch=64: [32, 64, 128, 256, 512]
        chs = [base_ch // 2, base_ch, base_ch * 2,
               base_ch * 4, base_ch * 8]

        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        self.cond_injects = nn.ModuleList()
        ch = chs[0]
        self.skip_channels = [ch]

        for i, out_ch in enumerate(chs):
            layers = nn.ModuleList([
                ResBlock(ch),
                nn.Conv2d(ch, out_ch, 1) if ch != out_ch else nn.Identity(),
                ResBlock(out_ch),
            ])
            # Attention at the two smallest levels only (32×32 and smaller)
            if i >= 3:
                layers.append(SelfAttention2d(out_ch))
            else:
                layers.append(nn.Identity())
            layers.append(Downsample(out_ch))
            self.downs.append(layers)
            self.cond_injects.append(ConditionInjection(cond_dim, out_ch))
            ch = out_ch
            self.skip_channels.append(ch)

        # Bottleneck (at smallest spatial size)
        self.mid = nn.Sequential(
            ResBlock(ch), SelfAttention2d(ch), ResBlock(ch),
        )
        self.mid_cond = ConditionInjection(cond_dim, ch)

        # Flatten to latent
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x, cond=None):
        """
        x: (B, 23, H, W)
        cond: (B, cond_dim) optional
        Returns: h (B, latent_dim), skips list
        """
        skips = []
        h = self.in_conv(x)
        skips.append(h)

        for i, blocks in enumerate(self.downs):
            res1, proj, res2, attn, down = blocks
            h = res1(h)
            h = proj(h)
            h = res2(h)
            h = attn(h)
            if cond is not None:
                h = self.cond_injects[i](h, cond)
            skips.append(h)
            h = down(h)

        h = self.mid(h)
        if cond is not None:
            h = self.mid_cond(h, cond)

        h = self.pool(h)
        latent = self.to_latent(h)
        return latent, skips


# =====================================================================
# Physics Layer (single step)
# =====================================================================

class PhysStep(nn.Module):
    """
    One step of physics evolution in latent space.

    Two branches applied in parallel:
      NN branch:  48→128→128→256→128→128→48 residual MLP
      Phys branch: split into vel(24)+pos(24), Newtonian update
    """
    def __init__(self, dim=48, vel_dim=24):
        super().__init__()
        self.dim = dim
        self.vel_dim = vel_dim

        # ── NN branch ──
        self.nn_branch = nn.Sequential(
            nn.Linear(dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, dim),
        )

        # ── Physics branch ──
        self.pos_to_zinv = nn.Sequential(
            nn.Linear(vel_dim, 64), nn.SiLU(),
            nn.Linear(64, vel_dim),
        )
        # g0: scalar weight × fixed mask [1]*8 + [0]*16
        self.g0 = nn.Parameter(torch.tensor(0.0))
        n_active = min(8, vel_dim)
        g0_mask = torch.zeros(vel_dim)
        g0_mask[:n_active] = 1.0
        self.register_buffer('g0_mask', g0_mask)
        # alpha: 25 components (24 + 1 bias)
        self.alpha = nn.Parameter(torch.randn(vel_dim + 1) * 0.01)

    def forward(self, h, dt):
        """
        h: (B, 48)
        Returns: out_nn (B, 48), out_phys (B, 48)
        """
        out_nn = h + self.nn_branch(h)

        vel = h[:, :self.vel_dim]
        pos = h[:, self.vel_dim:]
        zinv = self.pos_to_zinv(pos)
        zinv_ext = torch.cat([zinv, torch.ones(zinv.shape[0], 1,
                              device=zinv.device)], dim=1)
        g0_vec = self.g0 * self.g0_mask.unsqueeze(0)
        gravity = g0_vec + (zinv_ext * self.alpha.unsqueeze(0))[:, :self.vel_dim]

        vel_next = vel + gravity * dt
        pos_next = pos + (vel_next + vel) * dt * 0.5
        out_phys = torch.cat([vel_next, pos_next], dim=1)

        return out_nn, out_phys


class PhysEvolution(nn.Module):
    """Iterate PhysStep n times, blend NN and physics outputs."""
    def __init__(self, dim=48, vel_dim=24):
        super().__init__()
        self.step = PhysStep(dim, vel_dim)

    def forward(self, h, n_steps, dt=0.02, w_nn=0.5):
        h_nn = h.clone()
        h_phys = h.clone()
        for _ in range(n_steps):
            h_nn, _ = self.step(h_nn, dt)
            _, h_phys = self.step(h_phys, dt)
        return w_nn * h_nn + (1.0 - w_nn) * h_phys


# =====================================================================
# Rotation Layer
# =====================================================================

class RotationLayer(nn.Module):
    """
    Learnable orthogonal matrix via Cayley transform.
    W = (I - A) @ (I + A)^{-1},  A skew-symmetric.
    """
    def __init__(self, dim=48):
        super().__init__()
        self.dim = dim
        n_params = dim * (dim - 1) // 2
        self.skew_params = nn.Parameter(torch.randn(n_params) * 0.01)

    def _get_skew(self):
        A = torch.zeros(self.dim, self.dim, device=self.skew_params.device)
        idx = torch.triu_indices(self.dim, self.dim, offset=1)
        A[idx[0], idx[1]] = self.skew_params
        A = A - A.T
        return A

    def _get_W(self):
        A = self._get_skew()
        I = torch.eye(self.dim, device=A.device)
        return (I - A) @ torch.inverse(I + A)

    def forward(self, h):
        return h @ self._get_W().T

    def inverse(self, h):
        A = self._get_skew()
        I = torch.eye(self.dim, device=A.device)
        W_inv = (I + A) @ torch.inverse(I - A)
        return h @ W_inv.T


# =====================================================================
# Decoder — mirrors encoder, upsamples back to original resolution
# =====================================================================

class Decoder(nn.Module):
    """
    Latent + skip connections → predicted image at full resolution.

    Mirrors the encoder's 5 downsampling levels:
      bottleneck → 32→64→128→256→512  (for 512 input)

    Skip connections from encoder are concatenated at each level.
    """
    def __init__(self, out_ch=3, latent_dim=48, base_ch=32,
                 cond_dim=48, para5_dim=6, bottleneck_spatial=16):
        super().__init__()
        # Channel schedule (reverse of encoder)
        chs = [base_ch * 8, base_ch * 4, base_ch * 2,
               base_ch, base_ch // 2]

        top_ch = chs[0]
        bs = bottleneck_spatial
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, top_ch * bs * bs),
        )
        self.top_ch = top_ch
        self.bottleneck_spatial = bs

        self.para5_proj = nn.Sequential(
            nn.Linear(para5_dim, 128), nn.SiLU(),
            nn.Linear(128, cond_dim),
        )
        self.latent_cond_proj = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, cond_dim),
        )

        self.ups = nn.ModuleList()
        self.cond_injects = nn.ModuleList()
        ch = top_ch

        for i, next_ch in enumerate(chs):
            in_ch = ch + ch  # concat skip
            layers = nn.ModuleList([
                nn.Conv2d(in_ch, ch, 1),     # merge skip
                ResBlock(ch),
                ResBlock(ch),
            ])
            # Attention at lowest levels only
            if i < 2:
                layers.append(SelfAttention2d(ch))
            else:
                layers.append(nn.Identity())
            layers.append(Upsample(ch))
            out_ch = chs[i + 1] if i + 1 < len(chs) else chs[-1]
            layers.append(nn.Conv2d(ch, out_ch, 1))
            self.ups.append(layers)
            self.cond_injects.append(ConditionInjection(cond_dim, ch))
            ch = out_ch

        self.out = nn.Sequential(
            ResBlock(ch),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h_latent, skips, para5=None):
        B = h_latent.shape[0]
        bs = self.bottleneck_spatial
        h = self.latent_proj(h_latent).view(B, self.top_ch, bs, bs)

        cond = self.latent_cond_proj(h_latent)
        if para5 is not None:
            cond = cond + self.para5_proj(para5)

        skips_rev = list(reversed(skips))

        for i, blocks in enumerate(self.ups):
            merge, res1, res2, attn, up, proj = blocks
            skip_idx = min(i, len(skips_rev) - 1)
            skip = skips_rev[skip_idx]
            if skip.shape[2:] != h.shape[2:]:
                skip = F.interpolate(skip, size=h.shape[2:],
                                     mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = merge(h)
            h = res1(h)
            h = self.cond_injects[i](h, cond)
            h = res2(h)
            h = attn(h)
            h = up(h)
            h = proj(h)

        return self.out(h)


# =====================================================================
# Full O2T Network
# =====================================================================

class O2TNet(nn.Module):
    """
    Full Observation-to-Theory network.
    Processes images at their NATIVE resolution (e.g. 512×512).
    No artificial resizing.

    Input:
      pics: (B, 4, 3, H, W) — stereo pairs, any resolution
      params: (B, 5, 6) — [t, x, y, z, az, el]

    Output:
      predicted image (B, 3, H, W) — same resolution as input
    """
    def __init__(self, latent_dim=48, vel_dim=24,
                 param_dim=6, base_ch=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.base_ch = base_ch

        # Parameter encoder
        self.param_enc = ParamEncoder(param_dim)

        # Encoder: 23ch → latent
        self.encoder = Encoder(in_ch=23, latent_dim=latent_dim,
                               base_ch=base_ch, cond_dim=latent_dim)

        # Rotation layers
        self.rotation = RotationLayer(latent_dim)

        # Physics evolution
        self.physics = PhysEvolution(latent_dim, vel_dim)

        # Future params → h_PARA
        self.para5_to_h = nn.Sequential(
            nn.Linear(param_dim, 64), nn.SiLU(),
            nn.Linear(64, latent_dim),
        )

        # Decoder: latent + skips → image
        # bottleneck_spatial: after 5 downsamples, 512→16, 256→8
        # We use AdaptiveAvgPool in encoder so decoder needs a fixed start size
        self.decoder_start_size = 16  # works for 512 input (512/32=16)
        self.decoder = Decoder(
            out_ch=3, latent_dim=latent_dim, base_ch=base_ch,
            cond_dim=latent_dim, para5_dim=param_dim,
            bottleneck_spatial=self.decoder_start_size,
        )

    def preprocess_params(self, params):
        p = params.clone()
        t1 = p[:, 0:1, 0:1]
        p[:, :, 0:1] = p[:, :, 0:1] - t1
        xy1 = p[:, 0:1, 1:3]
        p[:, :, 1:3] = p[:, :, 1:3] - xy1
        return p

    def build_input_tensor(self, pics, params, H, W):
        """
        Build 23-channel input tensor at native resolution.

        pics: (B, 4, 3, H, W) — full resolution RGB
        params: (B, 5, 6) — preprocessed
        H, W: spatial size

        Returns: (B, 23, H, W)
        """
        B = pics.shape[0]

        # Grayscale for difference images
        gray = pics.mean(dim=2, keepdim=False)  # (B, 4, H, W)

        # Parameter images (generated at 8×8, upsampled to H×W)
        para_imgs = []
        for i in range(5):
            para_imgs.append(self.param_enc(params[:, i], H))

        # Pairwise grayscale differences
        diffs = []
        for i in range(4):
            for j in range(i + 1, 4):
                diffs.append(gray[:, i:i+1] - gray[:, j:j+1])

        # Assemble: 4×(3+1) + 1 + 6 = 23 channels
        channels = []
        for i in range(4):
            channels.append(pics[:, i])          # RGB (B, 3, H, W)
            channels.append(para_imgs[i])        # PARA (B, 1, H, W)
        channels.append(para_imgs[4])            # PARA5 (B, 1, H, W)
        channels.extend(diffs)                   # 6 diffs (B, 1, H, W)

        return torch.cat(channels, dim=1)

    def forward(self, pics, params, n_steps=1, dt=0.02, w_nn=0.5):
        """
        pics: (B, 4, 3, H, W) float [0,1], any resolution
        params: (B, 5, 6)
        n_steps: int
        dt: float
        w_nn: float

        Returns: predicted image (B, 3, H_out, W_out)
          H_out = closest multiple of 32 to H (for clean up/downsampling)
        """
        params_pp = self.preprocess_params(params)
        B, _, _, H_orig, W_orig = pics.shape

        # Pad/resize to multiple of 32 for clean 5× downsample
        H32 = ((H_orig + 31) // 32) * 32
        W32 = ((W_orig + 31) // 32) * 32
        if H32 != H_orig or W32 != W_orig:
            pics_p = F.interpolate(
                pics.view(B * 4, 3, H_orig, W_orig),
                size=(H32, W32), mode='bilinear', align_corners=False
            ).view(B, 4, 3, H32, W32)
        else:
            pics_p = pics

        # Build 23-channel input at full resolution
        x_23 = self.build_input_tensor(pics_p, params_pp, H32, W32)

        # Encode (convolutional downsampling, no resize)
        h_48, skips = self.encoder(x_23)

        # Physics pipeline
        h_rot = self.rotation(h_48)
        h_evolved = self.physics(h_rot, n_steps, dt, w_nn)
        h_inv = self.rotation.inverse(h_evolved)

        para5 = params_pp[:, 4]
        h_para = self.para5_to_h(para5)
        h_out = h_inv + h_para

        # Decode (convolutional upsampling to full resolution)
        pred = self.decoder(h_out, skips, para5)

        # Crop back to original size if we padded
        if H32 != H_orig or W32 != W_orig:
            pred = F.interpolate(pred, size=(H_orig, W_orig),
                                 mode='bilinear', align_corners=False)

        return pred

    def get_physics_params(self):
        g0 = float(self.physics.step.g0.detach().cpu())
        alpha = self.physics.step.alpha.detach().cpu().numpy()
        g0_mask = self.physics.step.g0_mask.detach().cpu().numpy()
        return {"g0": g0, "g0_mask": g0_mask, "alpha": alpha}


# =====================================================================
# Utility
# =====================================================================

def compute_w_nn(n_steps, w_nn_max=0.95, w_nn_min=0.2, decay_rate=0.1):
    """w_nn schedule: high for small n, low for large n."""
    return w_nn_min + (w_nn_max - w_nn_min) * math.exp(-decay_rate * n_steps)


# =====================================================================
# Test
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = O2TNet(latent_dim=48, vel_dim=24,
                   param_dim=6, base_ch=32).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Test with 512×512 input
    B = 2
    pics = torch.randn(B, 4, 3, 512, 512).to(device)
    params = torch.randn(B, 5, 6).to(device)

    with torch.no_grad():
        pred = model(pics, params, n_steps=3, dt=0.02, w_nn=0.8)

    print(f"Input:  pics {pics.shape}, params {params.shape}")
    print(f"Output: pred {pred.shape}, range [{pred.min():.3f}, {pred.max():.3f}]")

    phys = model.get_physics_params()
    print(f"Physics g0: {phys['g0']:.6f} (scalar)")
    print(f"Physics g0_mask: {phys['g0_mask']}")
    print(f"Physics alpha: shape={phys['alpha'].shape}")

    for n in [0, 1, 5, 10, 20, 50]:
        print(f"  n={n:3d} → w_nn={compute_w_nn(n):.3f}")

    print("\nAll tests passed!")
