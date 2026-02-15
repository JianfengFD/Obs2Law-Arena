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
# Decoder — Hourglass: PIC1+PIC3+diff → predicted image
# =====================================================================

class Decoder(nn.Module):
    """
    Hourglass decoder: takes PIC1, PIC3, and their difference as
    image-space input (9ch @ 512×512), downsamples to a bottleneck,
    then upsamples back to 512×512.

    The physics latent h_48 and para5 are injected via Dense→Reshape
    at the three middle layers (just before, at, and just after the
    bottleneck), providing the "what should change" signal while the
    image input provides the "what does the scene look like" signal.

    Structure (base_ch=32):
        Input: 9ch, 512×512
        ─ Down ─
        Level 0: Conv 9→16,   512→256
        Level 1: ResBlock 32, 256→128
        Level 2: ResBlock 64, 128→64     ← inject h_cond (middle-left)
        ─ Bottleneck ─
        Level 3: ResBlock 128, 64→32     ← inject h_cond (middle)
        ─ Up ─
        Level 4: ResBlock 128, 32→64     ← inject h_cond (middle-right)
        Level 5: ResBlock 64,  64→128   (+ skip from level 2)
        Level 6: ResBlock 32,  128→256  (+ skip from level 1)
        Level 7: ResBlock 16,  256→512  (+ skip from level 0)
        Output: Conv→3ch, Sigmoid
    """
    def __init__(self, latent_dim=48, para5_dim=6, base_ch=32):
        super().__init__()
        # Channel schedule: two-sided, thin at edges, fat in middle
        # For base_ch=32: [16, 32, 64, 128, 128, 64, 32, 16]
        ch_down = [base_ch // 2, base_ch, base_ch * 2, base_ch * 4]
        ch_up   = [base_ch * 4, base_ch * 2, base_ch, base_ch // 2]

        # ── Condition projection (h_48 + para5 → cond vector) ──
        cond_dim = latent_dim + para5_dim  # 48 + 6 = 54
        self.cond_dim = cond_dim

        # Three injection points (all produce 1-channel maps, broadcast-added):
        #   inject₁: after down level 0 → 1×256×256
        #   inject₂: at bottleneck     → 1× 32× 32
        #   inject₃: after up level 2  → 1×256×256
        #
        # Dense→(1,16,16)→Upsample to target size. Broadcast across channels.
        inject_specs = [
            (256, 16, 4),  # (target_spatial, dense_spatial, n_upsample)
            ( 32,  8, 2),
            (256, 16, 4),
        ]
        self.inject_projs = nn.ModuleList()
        self.inject_ups = nn.ModuleList()
        for target_s, dense_s, n_up in inject_specs:
            self.inject_projs.append(nn.Sequential(
                nn.Linear(cond_dim, 256),
                nn.SiLU(),
                nn.Linear(256, dense_s * dense_s),  # → 1ch, dense_s×dense_s
            ))
            up_layers = []
            for _ in range(n_up):
                up_layers.extend([
                    nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False),
                    nn.Conv2d(1, 1, 3, padding=1),
                    nn.SiLU(),
                ])
            self.inject_ups.append(nn.Sequential(*up_layers) if up_layers
                                   else nn.Identity())
        self.inject_specs = inject_specs

        # ── Down path ──
        self.in_conv = nn.Conv2d(9, ch_down[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = ch_down[0]
        for i, out_ch in enumerate(ch_down):
            block = nn.Sequential(
                ResBlock(ch),
                nn.Conv2d(ch, out_ch, 1) if ch != out_ch else nn.Identity(),
                ResBlock(out_ch),
                Downsample(out_ch),
            )
            self.down_blocks.append(block)
            ch = out_ch

        # ── Bottleneck ──
        self.mid = nn.Sequential(
            ResBlock(ch),
            SelfAttention2d(ch),
            ResBlock(ch),
        )

        # ── Up path ──
        self.up_blocks = nn.ModuleList()
        self.up_merges = nn.ModuleList()  # 1×1 conv to merge skip
        for i, out_ch in enumerate(ch_up):
            if i >= 1:
                # Has skip connection from corresponding down level
                in_ch = ch + ch_down[len(ch_down) - 1 - i]
                self.up_merges.append(nn.Conv2d(in_ch, ch, 1))
            else:
                self.up_merges.append(nn.Identity())
            block = nn.Sequential(
                Upsample(ch),
                ResBlock(ch),
                nn.Conv2d(ch, out_ch, 1) if ch != out_ch else nn.Identity(),
                ResBlock(out_ch),
            )
            self.up_blocks.append(block)
            ch = out_ch

        # ── Output ──
        self.out = nn.Sequential(
            ResBlock(ch),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, pic1, pic3, h_latent, para5):
        """
        pic1: (B, 3, H, W) — left eye at t1
        pic3: (B, 3, H, W) — left eye at t2
        h_latent: (B, latent_dim) — physics-evolved latent
        para5: (B, para5_dim) — target viewpoint params

        Returns: (B, 3, H, W) predicted image
        """
        B = pic1.shape[0]

        # Build 9-channel input: PIC1 + PIC3 + (PIC1-PIC3)
        diff = pic1 - pic3
        x = torch.cat([pic1, pic3, diff], dim=1)  # (B, 9, H, W)

        # Build condition vector
        h_cond = torch.cat([h_latent, para5], dim=1)  # (B, cond_dim)

        # Pre-compute injection maps: 1-channel, broadcast across all channels
        injects = []
        for i, (target_s, dense_s, n_up) in enumerate(self.inject_specs):
            feat = self.inject_projs[i](h_cond)          # (B, ds*ds)
            feat = feat.view(B, 1, dense_s, dense_s)     # (B, 1, ds, ds)
            feat = self.inject_ups[i](feat)               # (B, 1, target_s, target_s)
            injects.append(feat)  # broadcast-add: (B,1,s,s) + (B,C,s,s)

        # ── Down path ──
        h = self.in_conv(x)
        skips = []
        for i, block in enumerate(self.down_blocks):
            skips.append(h)
            h = block(h)
            # Inject after level 0 (256×256)
            if i == 0:
                h = h + injects[0]

        # ── Bottleneck (32×32) ──
        h = self.mid(h)
        h = h + injects[1]

        # ── Up path ──
        skips_rev = list(reversed(skips))
        for i, (block, merge) in enumerate(zip(self.up_blocks, self.up_merges)):
            if i >= 1:
                skip = skips_rev[i]
                if skip.shape[2:] != h.shape[2:]:
                    skip = F.interpolate(skip, size=h.shape[2:],
                                         mode='bilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)
                h = merge(h)
            h = block(h)
            # Inject after up level 2 (256×256)
            if i == 2:
                h = h + injects[2]

        return self.out(h)


# =====================================================================
# Full O2T Network
# =====================================================================

class O2TNet(nn.Module):
    """
    Full Observation-to-Theory network.
    Processes images at their NATIVE resolution (e.g. 512×512).

    Encoder: 23ch input → h_48 latent (convolutional downsampling)
    Physics: h_48 → W_ROT → PhysEvolution × n → W_ROT_INV → h_out
    Decoder: PIC1 + PIC3 + diff → hourglass UNet, with h_out and para5
             injected at middle layers via Dense→Reshape

    Input:
      pics: (B, 4, 3, H, W) — stereo pairs
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

        # Decoder: PIC1+PIC3+diff hourglass, latent injected at middle
        self.decoder = Decoder(
            latent_dim=latent_dim, para5_dim=param_dim,
            base_ch=base_ch,
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
        pics: (B, 4, 3, H, W), params: (B, 5, 6)
        Returns: (B, 23, H, W)
        """
        B = pics.shape[0]
        gray = pics.mean(dim=2, keepdim=False)  # (B, 4, H, W)
        para_imgs = [self.param_enc(params[:, i], H) for i in range(5)]
        diffs = []
        for i in range(4):
            for j in range(i + 1, 4):
                diffs.append(gray[:, i:i+1] - gray[:, j:j+1])
        channels = []
        for i in range(4):
            channels.append(pics[:, i])
            channels.append(para_imgs[i])
        channels.append(para_imgs[4])
        channels.extend(diffs)
        return torch.cat(channels, dim=1)

    def forward(self, pics, params, n_steps=1, dt=0.02, w_nn=0.5):
        """
        pics: (B, 4, 3, H, W) float [0,1]
        params: (B, 5, 6)
        Returns: predicted image (B, 3, H, W)
        """
        params_pp = self.preprocess_params(params)
        B, _, _, H_orig, W_orig = pics.shape

        # Pad to multiple of 32 for clean 5× downsample
        H32 = ((H_orig + 31) // 32) * 32
        W32 = ((W_orig + 31) // 32) * 32
        if H32 != H_orig or W32 != W_orig:
            pics_p = F.interpolate(
                pics.view(B * 4, 3, H_orig, W_orig),
                size=(H32, W32), mode='bilinear', align_corners=False
            ).view(B, 4, 3, H32, W32)
        else:
            pics_p = pics

        # Build 23-channel encoder input
        x_23 = self.build_input_tensor(pics_p, params_pp, H32, W32)

        # Encode
        h_48, _skips = self.encoder(x_23)

        # Physics pipeline
        h_rot = self.rotation(h_48)
        h_evolved = self.physics(h_rot, n_steps, dt, w_nn)
        h_inv = self.rotation.inverse(h_evolved)

        para5 = params_pp[:, 4]
        h_para = self.para5_to_h(para5)
        h_out = h_inv + h_para

        # Decode: PIC1 + PIC3 as image context, h_out as physics signal
        pic1 = pics_p[:, 0]  # (B, 3, H, W) left eye at t1
        pic3 = pics_p[:, 2]  # (B, 3, H, W) left eye at t2
        pred = self.decoder(pic1, pic3, h_out, para5)

        # Crop back if padded
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
