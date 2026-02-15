#!/usr/bin/env python3
"""
train_o2t.py — O2T Training with Replay Buffer
===================================================
CPU workers continuously render 512×512 samples into a shared pool.
GPU (or CPU) trains from the pool. Pool samples are gradually replaced.

Architecture:
    ┌──────────────┐          ┌──────────────┐
    │  CPU Worker 1 │─┐       │              │
    │  CPU Worker 2 │─┼──────▶│ Replay Buffer│──────▶ GPU Training
    │  ...          │─┤       │ (5000 samples)│
    │  CPU Worker N │─┘       │              │
    └──────────────┘          └──────────────┘

Usage:
    # Standard training (auto-detects MPS/CUDA/CPU)
    python train_o2t.py --module render_scene_a1e

    # Custom settings
    python train_o2t.py --module render_scene_a1e \
        --pool_size 5000 --render_workers 6 \
        --batch 16 --base_ch 32 --train_steps 50000

    # CPU-only debugging
    python train_o2t.py --module render_scene_a1e \
        --device cpu --batch 2 --pool_size 200 --render_workers 2

    # Resume
    python train_o2t.py --module render_scene_a1e --resume o2t_runs/ckpt_5000.pt
"""

import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import sys, argparse, importlib, time, math, json
import contextlib, io, threading, queue
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from o2t_net import O2TNet, compute_w_nn


# =====================================================================
# Stdout suppression for renderer
# =====================================================================

@contextlib.contextmanager
def suppress_stdout():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


# =====================================================================
# Sample Rendering (single sample)
# =====================================================================

def render_one(render_fn, seed, t, az, el, obs, img_size):
    with suppress_stdout():
        imgs = render_fn(
            seed_setup=seed, times=float(t),
            angles=(float(az), float(el)),
            observer_pos=tuple(float(x) for x in obs),
            image_size=(img_size, img_size))
    img = imgs[0] if isinstance(imgs, list) else imgs
    return img.astype(np.float32) / 255.0


def has_motion(render_fn, seed, t1, t2, az, el, obs, img_size):
    if abs(t2 - t1) < 1e-6:
        return True
    try:
        i1 = render_one(render_fn, seed, t1, az, el, obs, img_size)
        i2 = render_one(render_fn, seed, t2, az, el, obs, img_size)
        return np.abs(i1 - i2).mean() > 0.008
    except:
        return False


def make_stereo_obs(az, oc, gap=0.2):
    az_rad = np.radians(az)
    px, py = -np.sin(az_rad), np.cos(az_rad)
    h = gap / 2.0
    L = [oc[0]+px*h, oc[1]+py*h, oc[2]]
    R = [oc[0]-px*h, oc[1]-py*h, oc[2]]
    return L, R


def generate_sample(render_fn, mode, n_steps, rng, img_size=512, delta_t=0.02):
    """Generate one training sample. Returns dict or None."""
    for _ in range(10):
        seed = rng.randint(0, 100000)
        az, el = rng.uniform(0, 360), rng.uniform(-10, 30)
        oz = rng.uniform(1.5, 3.0)
        oc = [rng.uniform(-2, 2), rng.uniform(-2, 2), oz]
        t1 = rng.uniform(0, 5.0)

        if mode == 'DIS_MOD':
            t2 = t1
            tt, ta, te = t1, az, el
            to_ = list(oc)
        elif mode == 'VIEW_MOD':
            t2 = t1 + rng.choice([0, 1]) * delta_t
            tt, ta = t1, az + rng.uniform(-15, 15)
            te = el + rng.uniform(-5, 5)
            to_ = [oc[0]+rng.uniform(-1, 1), oc[1]+rng.uniform(-1, 1), oz]
        else:  # FUTURE_MOD
            t2 = t1 + rng.choice([0, 1]) * delta_t
            tt = t1 + n_steps * delta_t
            ta, te = az + rng.uniform(-10, 10), el + rng.uniform(-3, 3)
            to_ = [oc[0]+rng.uniform(-.5, .5), oc[1]+rng.uniform(-.5, .5), oz]

        if mode == 'FUTURE_MOD':
            if not has_motion(render_fn, seed, t1, tt, az, el, tuple(oc), img_size):
                continue

        try:
            oL1, oR1 = make_stereo_obs(az, oc)
            oL2, oR2 = make_stereo_obs(az, oc)
            pic1 = render_one(render_fn, seed, t1, az, el, oL1, img_size)
            pic2 = render_one(render_fn, seed, t1, az, el, oR1, img_size)
            pic3 = render_one(render_fn, seed, t2, az, el, oL2, img_size)
            pic4 = render_one(render_fn, seed, t2, az, el, oR2, img_size)
            tgt  = render_one(render_fn, seed, tt, ta, te, to_, img_size)

            def mkp(t, o, a, e):
                return np.array([t, o[0], o[1], o[2], a, e], np.float32)

            pics = np.stack([pic1, pic2, pic3, pic4]).transpose(0, 3, 1, 2)
            params = np.stack([mkp(t1,oL1,az,el), mkp(t1,oR1,az,el),
                               mkp(t2,oL2,az,el), mkp(t2,oR2,az,el),
                               mkp(tt,to_,ta,te)])
            target = tgt.transpose(2, 0, 1)

            return {"pics": pics, "params": params, "target": target,
                    "mode": mode, "n_steps": n_steps}
        except:
            continue
    return None


# =====================================================================
# Training Schedule
# =====================================================================

class TrainingSchedule:
    """
    Curriculum:
      Step 0–1K:        [1.0, 0, 0]   DIS only
      Step 1K–10K:      [0.5, 0.5, 0]  DIS + VIEW
      Step 10K–50K:     [0.2, 0.5, 0.3] + FUTURE (n=1-2)
      Step 50K+:        [0.1, 0.1, 0.8] heavy FUTURE (n=1-20)
    """
    def __init__(self):
        self.phases = [
            (1000,  [1.0, 0.0, 0.0]),
            (10000, [0.5, 0.5, 0.0]),
            (50000, [0.2, 0.5, 0.3]),
            (1e9,   [0.1, 0.1, 0.8]),
        ]

    def get_mode_and_n(self, step, rng):
        probs = self.phases[-1][1]
        for thr, p in self.phases:
            if step < thr:
                probs = p
                break
        idx = rng.choice(3, p=probs)
        mode = ['DIS_MOD', 'VIEW_MOD', 'FUTURE_MOD'][idx]
        if mode != 'FUTURE_MOD':
            return mode, 0
        if step < 50000:
            return mode, rng.choice([1, 2])
        progress = min(1.0, (step - 50000) / 450000)
        mu = 3 + progress * 12
        sig = 1 + progress * 7
        return mode, max(1, min(int(round(rng.normal(mu, sig))), 50))


# =====================================================================
# Replay Buffer
# =====================================================================

class ReplayBuffer:
    """
    Thread-safe replay buffer with gradual replacement.

    Stores samples as numpy arrays. Training loop samples random
    mini-batches. CPU workers continuously add new samples,
    replacing the oldest ones.
    """
    def __init__(self, capacity, img_size=512):
        self.capacity = capacity
        self.img_size = img_size
        self.lock = threading.Lock()

        # Pre-allocate arrays (filled with zeros initially)
        S = img_size
        self.pics = np.zeros((capacity, 4, 3, S, S), dtype=np.float32)
        self.params = np.zeros((capacity, 5, 6), dtype=np.float32)
        self.targets = np.zeros((capacity, 3, S, S), dtype=np.float32)
        self.n_steps = np.zeros(capacity, dtype=np.int32)
        self.modes = np.array(['DIS_MOD'] * capacity)

        self.size = 0          # how many valid samples
        self.write_idx = 0     # next position to write
        self.total_added = 0   # total samples ever added

    def add(self, sample):
        """Add one sample dict to the buffer."""
        with self.lock:
            idx = self.write_idx
            self.pics[idx] = sample["pics"]
            self.params[idx] = sample["params"]
            self.targets[idx] = sample["target"]
            self.n_steps[idx] = sample["n_steps"]
            self.modes[idx] = sample["mode"]
            self.write_idx = (self.write_idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self.total_added += 1

    def sample_batch(self, batch_size, rng):
        """Sample a random mini-batch. Returns numpy arrays."""
        with self.lock:
            if self.size < batch_size:
                indices = rng.choice(self.size, batch_size, replace=True)
            else:
                indices = rng.choice(self.size, batch_size, replace=False)
            return {
                "pics": self.pics[indices].copy(),
                "params": self.params[indices].copy(),
                "targets": self.targets[indices].copy(),
                "n_steps": int(self.n_steps[indices].max()),  # max n in batch
            }

    def fill_initial(self, render_fn, n_samples, schedule, rng,
                     img_size=512, delta_t=0.02):
        """Fill the buffer with initial samples (blocking, single-thread)."""
        print(f"  Filling replay buffer with {n_samples} initial samples ...")
        t0 = time.time()
        count = 0
        while count < n_samples:
            mode, n = schedule.get_mode_and_n(0, rng)  # step=0 → DIS_MOD
            sample = generate_sample(render_fn, mode, n, rng, img_size, delta_t)
            if sample is not None:
                self.add(sample)
                count += 1
                if count % 50 == 0:
                    elapsed = time.time() - t0
                    rate = count / elapsed
                    eta = (n_samples - count) / max(rate, 0.01)
                    print(f"\r    [{count}/{n_samples}] "
                          f"{rate:.1f}/s  ETA: {eta:.0f}s", end="", flush=True)
        elapsed = time.time() - t0
        print(f"\n    Done: {n_samples} samples in {elapsed:.0f}s "
              f"({n_samples/elapsed:.1f}/s)")


# =====================================================================
# Background Renderer Thread
# =====================================================================

class BackgroundRenderer:
    """
    Runs CPU rendering in background threads, continuously
    adding new samples to the replay buffer.
    """
    def __init__(self, buffer, render_fn, schedule, n_workers=4,
                 img_size=512, delta_t=0.02):
        self.buffer = buffer
        self.render_fn = render_fn
        self.schedule = schedule
        self.n_workers = n_workers
        self.img_size = img_size
        self.delta_t = delta_t
        self.running = False
        self.threads = []
        self.current_step = 0  # updated by training loop
        self._lock = threading.Lock()

    def set_step(self, step):
        with self._lock:
            self.current_step = step

    def _worker(self, worker_id):
        """Worker loop: generate samples and add to buffer."""
        rng = np.random.RandomState(worker_id * 1000 + int(time.time()) % 10000)
        while self.running:
            with self._lock:
                step = self.current_step
            mode, n = self.schedule.get_mode_and_n(step, rng)
            sample = generate_sample(
                self.render_fn, mode, n, rng, self.img_size, self.delta_t)
            if sample is not None:
                self.buffer.add(sample)

    def start(self):
        self.running = True
        for i in range(self.n_workers):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)
        print(f"  Started {self.n_workers} background render workers")

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=5)
        self.threads = []


# =====================================================================
# Training Loop
# =====================================================================

def train(args):
    # Device
    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load renderer
    mod_name = args.module
    func_id = mod_name.replace("render_scene_", "")
    sys.path.insert(0, ".")
    mod = importlib.import_module(mod_name)
    render_fn = getattr(mod, f"render_scene_{func_id}")
    print(f"Renderer: {mod_name}")

    # Model
    model = O2TNet(
        latent_dim=args.latent_dim,
        vel_dim=args.latent_dim // 2, param_dim=6, base_ch=args.base_ch,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params*4/1024/1024:.1f} MB)")
    print(f"  render_size={args.render_size} (native, no resize)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_steps)
    loss_mse_fn = nn.MSELoss()
    loss_l1_fn = nn.L1Loss()
    schedule = TrainingSchedule()

    # Replay buffer
    buffer = ReplayBuffer(args.pool_size, img_size=args.render_size)
    rng = np.random.RandomState(args.seed)

    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    # Fill initial pool
    init_fill = min(args.initial_fill, args.pool_size)
    buffer.fill_initial(render_fn, init_fill, schedule, rng,
                        img_size=args.render_size, delta_t=args.delta_t)

    # Start background renderers
    bg = BackgroundRenderer(
        buffer, render_fn, schedule,
        n_workers=args.render_workers,
        img_size=args.render_size,
        delta_t=args.delta_t,
    )
    bg.start()

    # Logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train_log.jsonl")
    log_f = open(log_path, "a")

    print(f"\nTraining: {args.train_steps} steps, batch={args.batch}, "
          f"pool={args.pool_size}")
    print(f"Logging to {log_path}")
    print("=" * 60)

    model.train()
    t0 = time.time()
    running_loss = 0.0

    try:
        for step in range(start_step, args.train_steps):
            bg.set_step(step)

            # Sample from buffer
            batch = buffer.sample_batch(args.batch, rng)
            pics = torch.from_numpy(batch["pics"]).to(device)
            params = torch.from_numpy(batch["params"]).to(device)
            targets = torch.from_numpy(batch["targets"]).to(device)
            n_steps = batch["n_steps"]
            w_nn = compute_w_nn(n_steps)

            # Forward + backward
            optimizer.zero_grad()
            pred = model(pics, params, n_steps=n_steps,
                         dt=args.delta_t, w_nn=w_nn)
            loss = 0.7 * loss_mse_fn(pred, targets) + \
                   0.3 * loss_l1_fn(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            # Progress
            if (step + 1) % 10 == 0:
                print(f"\r  [{step+1}/{args.train_steps}] "
                      f"loss={loss.item():.4f}  "
                      f"pool={buffer.size}/{buffer.capacity} "
                      f"(+{buffer.total_added})",
                      end="", flush=True)

            # Log
            if (step + 1) % args.log_every == 0:
                avg = running_loss / args.log_every
                elapsed = time.time() - t0
                phys = model.get_physics_params()
                lr = optimizer.param_groups[0]["lr"]

                entry = {
                    "step": step + 1,
                    "loss": round(avg, 6),
                    "g0": round(phys["g0"], 6),
                    "alpha_mean": round(float(phys["alpha"].mean()), 6),
                    "lr": round(lr, 8),
                    "pool_size": buffer.size,
                    "total_rendered": buffer.total_added,
                    "elapsed_s": round(elapsed, 1),
                }
                log_f.write(json.dumps(entry) + "\n")
                log_f.flush()
                print(f"\n  step {step+1:>6d}  avg_loss={avg:.4f}  "
                      f"g0={phys['g0']:.4f}  "
                      f"α={float(phys['alpha'].mean()):.4f}  "
                      f"lr={lr:.2e}  "
                      f"pool=+{buffer.total_added}  ({elapsed:.0f}s)")
                running_loss = 0.0

            # Checkpoint
            if (step + 1) % args.save_every == 0:
                cp = os.path.join(args.log_dir, f"ckpt_{step+1}.pt")
                torch.save({
                    "step": step + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }, cp)
                print(f"  → Saved {cp}")

    finally:
        bg.stop()

    # Final save
    final = os.path.join(args.log_dir, "model_final.pt")
    torch.save({"step": args.train_steps,
                "model": model.state_dict(),
                "args": vars(args)}, final)
    phys = model.get_physics_params()
    print(f"\n\nDone. Model: {final}")
    print(f"  g0 = {phys['g0']:.6f}")
    print(f"  alpha = {phys['alpha']}")
    log_f.close()


def main():
    ap = argparse.ArgumentParser(
        description="O2T training with replay buffer")
    # Module
    ap.add_argument("--module", required=True,
                    help="Sealed render module, e.g. render_scene_a1e")
    # Training
    ap.add_argument("--train_steps", type=int, default=50000,
                    help="Total training steps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--delta_t", type=float, default=0.02)
    # Model
    ap.add_argument("--latent_dim", type=int, default=48)
    ap.add_argument("--base_ch", type=int, default=32,
                    help="Base channels (32→~15M, 64→~60M params)")
    # Rendering
    ap.add_argument("--render_size", type=int, default=512,
                    help="Render resolution (512 for full detail)")
    ap.add_argument("--render_workers", type=int, default=4,
                    help="Background CPU render threads")
    # Replay buffer
    ap.add_argument("--pool_size", type=int, default=5000,
                    help="Replay buffer capacity")
    ap.add_argument("--initial_fill", type=int, default=500,
                    help="Samples to pre-fill before training starts")
    # Logging
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_dir", default="o2t_runs")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None,
                    help="Force device (cpu/cuda/mps)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
