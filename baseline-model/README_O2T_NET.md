# O2T Neural Network — Architecture & Training Guide

## Overview

The O2T (Observation-to-Theory) network discovers physics laws from 512×512 rendered images. It has an explicit physics-informed bottleneck with learnable Newtonian parameters (g0, α) that converge to real gravity values during training.

**No artificial resizing** — the network processes 512×512 images natively through convolutional downsampling, preserving fine spatial detail critical for physics inference.

---

## Quick Start

```bash
python train_o2t.py --module render_scene_a1e
```

---

## Architecture Diagram

```
 ┌───────────────────────── ENCODER ──────────────────────────┐
 │                                                             │
 │  PIC1,2,3,4 (512×512)                                      │
 │  + PARA1..5 + diffs                                         │
 │       │                                                     │
 │    23ch, 512 → 256 → 128 → 64 → 32 → 16                   │
 │       │         conv downsampling (5 levels)                │
 │       ▼                                                     │
 │    AdaptiveAvgPool → Linear → h_48                          │
 │                                                             │
 └─────────────────────────────────────────────────────────────┘
                            │
                       ┌────▼────┐
                       │  W_ROT  │
                       └────┬────┘
                            │
                  ┌─────────▼──────────┐
                  │  PhysLayer × n     │
                  │  NN + Physics blend│
                  └─────────┬──────────┘
                            │
                       ┌────▼─────┐
                       │W_ROT_INV │
                       └────┬─────┘
                            │
                       + h_PARA(para5)
                            │
                         h_out (48-dim)
                            │
 ┌──────────────────── DECODER (hourglass) ───────────────────┐
 │                                                             │
 │   PIC1 + PIC3 + (PIC1−PIC3)                                │
 │        9ch, 512×512                                         │
 │        │                                                    │
 │     Down: 512→256→128→64→32                                 │
 │                   ↑        ↑                                │
 │              inject₁   inject₂  ← Dense(h_out ⊕ para5)     │
 │                         (bottleneck)                        │
 │     Up:   32→64→128→256→512                                 │
 │               ↑                                             │
 │          inject₃  ← Dense(h_out ⊕ para5)                   │
 │        │                                                    │
 │     Conv→3ch→Sigmoid → Predicted Image (512×512)            │
 │                                                             │
 └─────────────────────────────────────────────────────────────┘
```

---

## Encoder

Takes all 4 images + parameter maps + pairwise differences = 23 channels at 512×512.

```
23ch, 512×512
  → Conv 23→16
  → [ResBlock×2, Downsample]  → 16ch, 256×256
  → [ResBlock×2, Downsample]  → 32ch, 128×128
  → [ResBlock×2, Downsample]  → 64ch,  64×64
  → [ResBlock×2, Attn, Down]  → 128ch, 32×32
  → [ResBlock×2, Attn, Down]  → 256ch, 16×16
  → Bottleneck (ResBlock + SelfAttention + ResBlock)
  → AdaptiveAvgPool(1) → Linear(256→256→48) → h_48
```

Thin channels at high resolution (16ch @ 512, 32ch @ 256) keep memory manageable. Self-attention only at 32×32 and 16×16.

## Decoder (Hourglass)

The decoder is a **separate hourglass UNet** that takes two original images as input, not encoder skip connections. This lets it see the actual scene texture and structure directly.

**Input**: PIC1 (t1 left eye) + PIC3 (t2 left eye) + their difference = **9 channels at 512×512**

**Physics injection**: h_out (48-dim) and para5 (6-dim) are concatenated into a 54-dim condition vector, then projected via `Dense → 1ch 16×16 → Upsample×4 → 1ch 256×256` (or `→ 1ch 8×8 → Upsample×2 → 1ch 32×32` for the bottleneck). The resulting **single-channel** map is **broadcast-added** across all feature channels:

```
Input: 9ch, 512×512

─── Down path ───
Level 0: Conv 9→16,   Downsample → 256×256  ← inject₁ (1ch, 256×256, broadcast)
Level 1: ResBlock 32,  Downsample → 128×128
Level 2: ResBlock 64,  Downsample →  64×64
Level 3: ResBlock 128, Downsample →  32×32

─── Bottleneck ───
ResBlock + SelfAttention + ResBlock  @ 32×32  ← inject₂ (1ch, 32×32, broadcast)

─── Up path ───
Level 4: Upsample + ResBlock 128              →  64×64
Level 5: Upsample + ResBlock 64  (+ skip L2)  → 128×128
Level 6: Upsample + ResBlock 32  (+ skip L1)  → 256×256  ← inject₃ (1ch, 256×256, broadcast)
Level 7: Upsample + ResBlock 16  (+ skip L0)  → 512×512

Output: Conv 16→3, Sigmoid → predicted image [0,1]
```

The 1-channel broadcast design is lightweight (Dense output is only 256 floats for 16×16) while giving each spatial location a scalar "how much to change" signal that applies uniformly across all feature channels.

**Why this design?**
- The decoder **sees the actual scene** (PIC1, PIC3) — it knows what objects exist and where they are
- The latent h_out tells it **how the scene should change** (physics evolution + viewpoint shift)
- Dense→Reshape injection gives the latent full spatial expressiveness at the resolution where it matters (64×64 and 32×32)
- Skip connections within the decoder hourglass preserve texture detail
- The encoder's skip connections are NOT used — the two networks (encoder and decoder) communicate only through the 48-dim latent, forcing it to be physically meaningful

## Physics Evolution

Two parallel branches iterated n times:

**NN branch**: `48→128→…→48` residual MLP

**Physics branch**:
```
g0_vec = g0 · [1,1,1,1,1,1,1,1, 0,...,0]   (scalar × mask)
gravity = g0_vec + α(25) · [MLP(pos), 1]
vel += gravity · Δt
pos += (vel_old + vel_new) · Δt / 2
```

Blend: `w_nn · h_NN + (1−w_nn) · h_phys`, w_nn high for small n, low for large n.

## Rotation Layer

48×48 orthogonal matrix (Cayley transform, `torch.inverse`). Aligns encoder latent with physics vel/pos semantics.

---

## Training Modes & Curriculum

| Mode | n | Description |
|---|---|---|
| **DIS_MOD** | 0 | Reconstruct current image (autoencoder warmup) |
| **VIEW_MOD** | 0 | Predict different viewpoint, same time |
| **FUTURE_MOD** | ≥1 | Predict future scene state |

| Steps | DIS | VIEW | FUTURE | n range |
|---|---|---|---|---|
| 0–1K | 100% | — | — | — |
| 1K–10K | 50% | 50% | — | — |
| 10K–50K | 20% | 50% | 30% | 1–2 |
| 50K+ | 10% | 10% | 80% | 1–20 |

---

## Replay Buffer

CPU threads render 512×512 images continuously into a ring buffer. GPU trains from random samples.

| RAM | pool_size | initial_fill |
|---|---|---|
| 16 GB (M2) | 500 | 100 |
| 64 GB | 3000 | 500 |
| 128 GB+ | 5000 | 500 |

---

## Hardware Examples

```bash
# M2 MacBook (16 GB)
python train_o2t.py --module render_scene_a1e \
    --pool_size 500 --initial_fill 100 --render_workers 4 \
    --batch 2 --base_ch 32 --device mps

# GPU server (64 GB RAM, RTX 3090)
python train_o2t.py --module render_scene_a1e \
    --pool_size 3000 --render_workers 16 --batch 16 --base_ch 64

# Large server (128+ GB, 4× A100)
python train_o2t.py --module render_scene_a1e \
    --pool_size 5000 --render_workers 30 --batch 64 --base_ch 64
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
| `o2t_net.py` | Network (Encoder + hourglass Decoder + PhysLayer) |
| `train_o2t.py` | Training with replay buffer |

---

## Design Rationale

**Why does the decoder take PIC1+PIC3 instead of encoder skips?** The encoder's job is to compress 4 images into a physics-meaningful 48-dim latent. The decoder's job is to paint the future scene. By giving the decoder the actual images, it can directly copy textures and only modify what the latent says should change. This forces the latent to encode pure physics (positions, velocities, gravity), not scene appearance.

**Why hourglass (down then up) instead of just upsampling from latent?** Starting from a 16×16 feature map and upsampling to 512 requires generating all spatial detail from scratch. The hourglass starts from the actual 512 image and only needs to modify it — much easier to learn and produces sharper results.

**Why 1-channel broadcast injection?** A full multi-channel injection (e.g. 16ch × 256×256) would need a massive Dense layer or complex upsampler. A single channel broadcast-added to all feature channels is extremely lightweight (Dense produces just 256 values for a 16×16 seed) while still providing spatially-varying "how much to change here" signals. The feature channels already carry what-to-change semantics from the conv layers.

**Why inject at 256×256 (left/right) and 32×32 (center)?** The 256×256 injections bracket the down/up path at high resolution, giving the physics signal broad spatial influence early and late. The 32×32 bottleneck injection is where abstract scene understanding lives — object positions and motion directions. Together, the three points cover coarse-to-fine physics modulation.
