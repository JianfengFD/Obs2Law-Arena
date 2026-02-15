#!/usr/bin/env python3
"""
train_o2t_offline.py — Train O2T from Pre-generated Data
============================================================
Much faster than on-the-fly generation. Supports multi-GPU.

Usage:
    # Basic training (auto-detects stages in data dir)
    python train_o2t_offline.py --data_dir data_a1e

    # Resume from checkpoint
    python train_o2t_offline.py --data_dir data_a1e --resume o2t_runs/ckpt_5000.pt

    # Multi-GPU (PyTorch DDP)
    torchrun --nproc_per_node=4 train_o2t_offline.py --data_dir data_a1e --batch 64

    # Custom settings
    python train_o2t_offline.py --data_dir data_a1e --batch 32 --base_ch 64 --epochs_per_stage 10
"""

import os, sys, argparse, json, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from o2t_net import O2TNet, compute_w_nn


# =====================================================================
# Dataset
# =====================================================================

class O2TDataset(Dataset):
    """
    Load pre-generated .npz batch files into a flat dataset.
    Each .npz contains: pics (N,4,3,H,W), params (N,5,6),
                         targets (N,3,H,W), n_steps (N,), modes (N,)
    """
    def __init__(self, stage_dir):
        self.pics_list = []
        self.params_list = []
        self.targets_list = []
        self.n_steps_list = []

        npz_files = sorted(glob.glob(os.path.join(stage_dir, "batch_*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No batch_*.npz files in {stage_dir}")

        print(f"  Loading {len(npz_files)} batch files from {stage_dir} ...")
        for f in npz_files:
            data = np.load(f, allow_pickle=True)
            self.pics_list.append(data["pics"])
            self.params_list.append(data["params"])
            self.targets_list.append(data["targets"])
            self.n_steps_list.append(data["n_steps"])

        self.pics = np.concatenate(self.pics_list, axis=0)
        self.params = np.concatenate(self.params_list, axis=0)
        self.targets = np.concatenate(self.targets_list, axis=0)
        self.n_steps = np.concatenate(self.n_steps_list, axis=0)

        print(f"  Loaded {len(self)} samples, "
              f"n_steps range: [{self.n_steps.min()}, {self.n_steps.max()}]")

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.pics[idx]),
                torch.from_numpy(self.params[idx]),
                torch.from_numpy(self.targets[idx]),
                int(self.n_steps[idx]))


def collate_fn(batch):
    """Custom collate: group by n_steps for efficient physics iteration."""
    pics = torch.stack([b[0] for b in batch])
    params = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    n_steps = max(b[3] for b in batch)
    return pics, params, targets, n_steps


# =====================================================================
# Training
# =====================================================================

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, log_f):
    model.train()
    loss_mse_fn = nn.MSELoss()
    loss_l1_fn = nn.L1Loss()
    running_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (pics, params, targets, n_steps) in enumerate(loader):
        pics = pics.to(device)
        params = params.to(device)
        targets = targets.to(device)
        w_nn = compute_w_nn(n_steps)

        optimizer.zero_grad()
        pred = model(pics, params, n_steps=n_steps, dt=0.02, w_nn=w_nn)
        loss = 0.7 * loss_mse_fn(pred, targets) + 0.3 * loss_l1_fn(pred, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 20 == 0:
            avg = running_loss / n_batches
            elapsed = time.time() - t0
            phys = model.get_physics_params()
            print(f"\r  epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"loss={avg:.4f}  g0={phys['g0']:.4f}  "
                  f"α={float(phys['alpha'].mean()):.4f}  "
                  f"({elapsed:.0f}s)", end="", flush=True)

    avg_loss = running_loss / max(n_batches, 1)
    elapsed = time.time() - t0
    phys = model.get_physics_params()

    print(f"\n  epoch {epoch} done: avg_loss={avg_loss:.4f}  "
          f"g0={phys['g0']:.4f}  α={float(phys['alpha'].mean()):.4f}  "
          f"({elapsed:.0f}s)")

    if log_f:
        entry = {"epoch": epoch, "loss": round(avg_loss, 6),
                 "g0": round(phys["g0"], 6),
                 "alpha_mean": round(float(phys["alpha"].mean()), 6),
                 "time_s": round(elapsed, 1)}
        log_f.write(json.dumps(entry) + "\n")
        log_f.flush()

    return avg_loss


def train(args):
    # Device setup
    device = torch.device("mps" if hasattr(torch.backends, "mps")
                          and torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # Find stages
    data_dir = args.data_dir
    stage_dirs = []
    for sid in [1, 2, 3, 4]:
        stage_info = None
        for name in [f"stage{sid}_*", f"stage{sid}"]:
            matches = glob.glob(os.path.join(data_dir, name))
            if matches:
                stage_info = sorted(matches)[0]
                break
        if stage_info and os.path.isdir(stage_info):
            meta_path = os.path.join(stage_info, "meta.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            stage_dirs.append({
                "id": sid, "dir": stage_info, "meta": meta,
                "epochs": meta.get("epochs_recommended",
                                   args.epochs_per_stage),
            })

    if not stage_dirs:
        print(f"No stage directories found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(stage_dirs)} stages:")
    for s in stage_dirs:
        print(f"  Stage {s['id']}: {s['dir']}  "
              f"({s['meta'].get('n_samples', '?')} samples, "
              f"{s['epochs']} epochs)")

    # Model (native resolution, no img_size needed)
    model = O2TNet(
        latent_dim=args.latent_dim,
        vel_dim=args.latent_dim // 2,
        param_dim=6,
        base_ch=args.base_ch,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params*4/1024/1024:.1f} MB)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Resume
    start_stage = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_stage = ckpt.get("stage", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed: stage {start_stage+1}, epoch {start_epoch}")

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train_log.jsonl")
    log_f = open(log_path, "a")

    print("=" * 60)
    global_t0 = time.time()
    total_epoch = 0

    # Stage-by-stage training
    for stage_idx, stage_info in enumerate(stage_dirs):
        if stage_idx < start_stage:
            continue

        sid = stage_info["id"]
        print(f"\n{'='*60}")
        print(f"STAGE {sid}: {stage_info['meta'].get('description', '')}")
        print(f"{'='*60}")

        # Load dataset
        dataset = O2TDataset(stage_info["dir"])
        loader = DataLoader(
            dataset, batch_size=args.batch, shuffle=True,
            num_workers=min(4, args.batch), collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        # Compute total steps for scheduler
        n_epochs = stage_info["epochs"]
        if args.epochs_per_stage:
            n_epochs = args.epochs_per_stage
        total_steps = len(loader) * n_epochs

        # Reset scheduler per stage (warm restarts effectively)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps)

        # Reduce LR for later stages
        if sid >= 3:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * 0.5
        if sid >= 4:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * 0.2

        for epoch in range(n_epochs):
            if stage_idx == start_stage and epoch < start_epoch:
                continue

            total_epoch += 1
            avg_loss = train_one_epoch(
                model, loader, optimizer, scheduler, device,
                total_epoch, log_f)

            # Checkpoint every epoch
            ckpt_path = os.path.join(
                args.log_dir, f"ckpt_s{sid}_e{epoch+1}.pt")
            torch.save({
                "stage": stage_idx, "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "loss": avg_loss,
            }, ckpt_path)

        # Print physics after each stage
        phys = model.get_physics_params()
        print(f"\n  Stage {sid} complete. g0={phys['g0']:.6f}, "
              f"α_mean={float(phys['alpha'].mean()):.6f}")

    # Final save
    final_path = os.path.join(args.log_dir, "model_final.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, final_path)
    phys = model.get_physics_params()
    elapsed = time.time() - global_t0
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.0f}s")
    print(f"Final model: {final_path}")
    print(f"  g0 = {phys['g0']:.6f}")
    print(f"  alpha = {phys['alpha']}")
    log_f.close()


def main():
    ap = argparse.ArgumentParser(description="Train O2T from pre-generated data")
    ap.add_argument("--data_dir", required=True,
                    help="Directory with stage1_*, stage2_*, etc.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--latent_dim", type=int, default=48)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs_per_stage", type=int, default=None,
                    help="Override epochs per stage (default: from meta.json)")
    ap.add_argument("--log_dir", default="o2t_runs")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
