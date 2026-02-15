# O2T Neural Network — Architecture & Training Guide

## Overview

The O2T (Observation-to-Theory) network discovers physics laws from 512×512 rendered images. It has an explicit physics-informed bottleneck with learnable Newtonian parameters (g0, α) that are forced to converge to real gravity values during training.

Key principle: **no artificial resizing**. The network processes 512×512 images natively through convolutional downsampling, preserving fine spatial detail that is critical for tracking small objects and extracting physics.

---

## Quick Start

```bash
# Standard training (CPU renders 512×512, GPU trains)
python train_o2t.py --module render_scene_a1e

# M2 Mac (16 GB)
python train_o2t.py --module render_scene_a1e \
    --pool_size 500 --initial_fill 100 --render_workers 4 \
    --batch 4 --device mps

# GPU server (64+ GB RAM)
python train_o2t.py --module render_scene_a1e \
    --pool_size 3000 --render_workers 16 --batch 32 --base_ch 64
```

---

## Training Architecture

```
┌────────────────┐         ┌────────────────────┐        ┌──────────────┐
│ CPU Worker 1   │──┐      │                    │        │              │
│ CPU Worker 2   │──┼─────▶│   Replay Buffer    │───────▶│  GPU / CPU   │
│ ...            │──┤      │   (5000 samples    │        │  Training    │
│ CPU Worker N   │──┘      │    @ 512×512)       │        │              │
└────────────────┘         └────────────────────┘        └──────────────┘
  Continuously render        FIFO replacement              Sample random
  new 512×512 samples        oldest ← newest               mini-batches
```

The buffer pre-fills with `initial_fill` samples (default 500), then background CPU threads continuously render and replace. Training and rendering run simultaneously.

---

## Network Architecture

```
INPUT (512×512)                LATENT SPACE                    OUTPUT (512×512)
───────────────                ────────────                    ───────────────

PIC1 (t1, L) ─┐
PIC2 (t1, R) ─┤   ┌─────────────────────────────┐
PIC3 (t2, L) ─┼─ 23ch ─┤ Encoder (conv downsample)  ├─→ h_48
PIC4 (t2, R) ─┤   │ 512→256→128→64→32 (×½ each)  │
PARA 1..5 ─────┘   └─────────────────────────────┘     │
                                                        ▼
                                                   ┌────────┐
                                                   │ W_ROT  │
                                                   └───┬────┘
                                                       │
                                             ┌─────────▼──────────┐
                                             │  PhysLayer × n     │
                                             │  NN branch + Phys  │
                                             └─────────┬──────────┘
                                                       │
                                                  ┌────▼─────┐
                                                  │W_ROT_INV │
                                                  └────┬─────┘
                                                       │
                                                + h_PARA(para5)
                                                       │
 Encoder skips (512,256,128,64,32) ──────────→ ┌──────▼──────┐
                                               │   Decoder    │──→ 512×512
                                               │ 32→64→128→  │
                                               │ 256→512      │
                                               └─────────────┘
```

### Why no resize?

At 512×512, a ball has ~20-40px diameter with clearly visible trajectories. Resizing to 128×128 shrinks it to 5-10px — losing the subtle motion cues that physics inference depends on. Instead, the encoder **learns** what information to preserve through its convolutional filters, extracting position, velocity, and depth signals at full resolution before compressing to the 48-dim latent.

### 23-channel input

```
Ch 1–3:   PIC1 (RGB)           Ch 4:  PIC1_PARA (embedded)
Ch 5–7:   PIC2 (RGB)           Ch 8:  PIC2_PARA
Ch 9–11:  PIC3 (RGB)           Ch 12: PIC3_PARA
Ch 13–15: PIC4 (RGB)           Ch 16: PIC4_PARA
Ch 17:    PIC5_PARA (future params, no image)
Ch 18–23: 6 grayscale pairwise differences (stereo + temporal)
```

### Encoder (5 downsampling levels)

```
23ch, 512×512
  → Conv 23→16
  → [ResBlock, ResBlock, Downsample]  → 16ch, 256×256
  → [ResBlock, ResBlock, Downsample]  → 32ch, 128×128
  → [ResBlock, ResBlock, Downsample]  → 64ch,  64×64
  → [ResBlock, ResBlock, Attn, Down]  → 128ch, 32×32
  → [ResBlock, ResBlock, Attn, Down]  → 256ch, 16×16
  → Bottleneck: ResBlock + Attention + ResBlock
  → AdaptiveAvgPool(1) → Linear(256→256→48) → h_48
```

The first two levels use thin channels (16, 32) to keep memory manageable at high resolution. Self-attention is only applied at 32×32 and 16×16 where token counts are reasonable (1024 and 256).

### Physics Evolution (n iterations)

Two parallel branches, iterated n times:

**NN branch**: `48→128→128→256→128→128→48` residual MLP.

**Physics branch**:
```
vel(24), pos(24) = split(h_48)
g0_vec = g0 · [1,1,1,1,1,1,1,1, 0,...,0]   (scalar × mask)
gravity = g0_vec + α(25) · [MLP(pos), 1]
vel_next = vel + gravity · Δt
pos_next = pos + (vel + vel_next) · Δt / 2
```

Blend: `h_out = w_nn · h_NN + (1−w_nn) · h_phys`

| n | w_nn | Regime |
|---|---|---|
| 0 | 0.95 | NN only (reconstruction) |
| 5 | 0.56 | Balanced |
| 10+ | 0.22–0.35 | Physics dominant |

### Decoder (5 upsampling levels)

Mirrors the encoder with skip connections at each level:
```
h_48 → Linear → 256ch, 16×16
  → [Skip concat, Conv, ResBlock×2, Attn, Upsample] → 128ch, 32×32
  → [Skip concat, Conv, ResBlock×2, Upsample]        → 64ch, 64×64
  → [Skip concat, Conv, ResBlock×2, Upsample]        → 32ch, 128×128
  → [Skip concat, Conv, ResBlock×2, Upsample]        → 16ch, 256×256
  → [Skip concat, Conv, ResBlock×2, Upsample]        → 16ch, 512×512
  → Conv(16→3) → Sigmoid → predicted image [0,1]
```

### Rotation Layer (W_ROT)

48×48 orthogonal matrix via Cayley transform. 1,128 learnable parameters. Uses `torch.inverse` for MPS compatibility.

---

## Three Training Modes

| Mode | n | Viewpoint shift | Purpose |
|---|---|---|---|
| **DIS_MOD** | 0 | None | Image reconstruction (autoencoder warmup) |
| **VIEW_MOD** | 0 | Yes | 3D understanding (disentangle camera from physics) |
| **FUTURE_MOD** | ≥1 | Optional | Physics prediction (g0, α must learn real gravity) |

### Curriculum

| Step range | DIS | VIEW | FUTURE | n range |
|---|---|---|---|---|
| 0 – 1K | 100% | — | — | — |
| 1K – 10K | 50% | 50% | — | — |
| 10K – 50K | 20% | 50% | 30% | 1–2 |
| 50K+ | 10% | 10% | 80% | 1–20 (Gaussian) |

The curriculum governs both training targets and background sample generation. As training advances, the buffer is gradually filled with physics-heavy samples.

---

## Replay Buffer

Fixed-size ring buffer. CPU threads generate new samples; oldest samples are replaced FIFO.

### Memory usage (512×512)

```
Per sample: 4 pics × 3 × 512² × 4B + 1 target × 3 × 512² × 4B ≈ 15.7 MB

Pool  500 → ~ 8 GB     (M2 Mac 16GB)
Pool 1000 → ~16 GB     (32 GB server)
Pool 3000 → ~47 GB     (64 GB server)
Pool 5000 → ~78 GB     (128+ GB server)
```

Training starts as soon as `initial_fill` is ready. Background workers keep rendering.

---

## Command Reference

```bash
python train_o2t.py --module render_scene_a1e \
    --train_steps 50000        # Total iterations
    --batch 16                 # Mini-batch size
    --lr 1e-4                  # Learning rate
    --base_ch 32               # Base channels (32→~15M, 64→~60M)
    --latent_dim 48            # Latent dimension
    --render_size 512          # Image resolution
    --pool_size 5000           # Buffer capacity
    --initial_fill 500         # Pre-fill count
    --render_workers 4         # CPU threads
    --device cuda              # Force device
    --resume ckpt.pt           # Resume
```

---

## Hardware Recommendations

### M2 MacBook Pro (16 GB)

```bash
python train_o2t.py --module render_scene_a1e \
    --pool_size 500 --initial_fill 100 --render_workers 4 \
    --batch 2 --base_ch 32 --device mps
```

### GPU Server (64 GB RAM, 1× RTX 3090)

```bash
python train_o2t.py --module render_scene_a1e \
    --pool_size 3000 --initial_fill 500 --render_workers 16 \
    --batch 16 --base_ch 64
```

### Large GPU Server (128+ GB RAM, 4× A100)

```bash
python train_o2t.py --module render_scene_a1e \
    --pool_size 5000 --initial_fill 500 --render_workers 30 \
    --batch 64 --base_ch 64
```

---

## Inspecting Learned Physics

```python
import torch
from o2t_net import O2TNet

model = O2TNet(latent_dim=48, base_ch=32)
ckpt = torch.load("o2t_runs/model_final.pt", weights_only=False)
model.load_state_dict(ckpt["model"])

phys = model.get_physics_params()
print(f"g0 = {phys['g0']:.4f}")       # should → ~9.8
print(f"alpha = {phys['alpha']}")
```

---

## File Reference

| File | Description |
|---|---|
| `o2t_net.py` | Network architecture (native 512, no resize) |
| `train_o2t.py` | Training with replay buffer (CPU render + GPU train) |

---

## Design Rationale

**Why 512×512 natively?** Physics inference requires tracking objects across frames. At low resolution, balls become a few pixels — impossible to extract velocity or trajectory. The encoder's convolutional filters learn what matters at full resolution, then compress it. Resizing before the network discards information before the network has a chance to decide what's important.

**Why thin early channels?** 512×512 × 64ch would need ~64 MB per feature map per batch sample. Using 16ch at 512 and 32ch at 256 keeps memory under control while still capturing spatial detail. The channels widen as spatial size shrinks.

**Why replay buffer?** Three reasons: (1) the curriculum needs different sample types at different stages — continuous generation adapts; (2) CPU rendering is slow (~1s/image) but GPU training is fast (~0.01s/step), so decoupling maximizes both; (3) fresh samples improve generalization versus training on a fixed dataset.

**Why scalar g0 with mask?** A free 24-dim g0 distributes gravity across all latent dimensions, making it uninterpretable. The mask forces the rotation layer to align specific dimensions with the height axis, so g0 directly reads out the gravitational constant.

**Why 5 downsample levels?** 512 / 2⁵ = 16. Self-attention at 16×16 (256 tokens) and 32×32 (1024 tokens) is computationally feasible. The bottleneck is small enough for AdaptiveAvgPool to produce a meaningful 48-dim latent.
