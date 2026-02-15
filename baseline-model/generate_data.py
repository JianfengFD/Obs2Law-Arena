#!/usr/bin/env python3
"""
generate_data.py — Pre-generate Training Data for O2T Network
================================================================
Generates training data in stages, saved as .npz files.
Supports multiprocessing for speed.

Usage:
    # Generate all stages for a module
    python generate_data.py --module render_scene_a1e --output_dir data_a1e

    # Generate a specific stage only
    python generate_data.py --module render_scene_a1e --output_dir data_a1e --stage 1

    # Use 8 parallel workers
    python generate_data.py --module render_scene_a1e --output_dir data_a1e --workers 8

    # Custom sample count
    python generate_data.py --module render_scene_a1e --output_dir data_a1e --samples_per_stage 5000

Data Layout:
    data_a1e/
        stage1_DIS/          2000 samples, mode=DIS_MOD, n=0
            batch_000.npz
            batch_001.npz
            ...
        stage2_DIS_VIEW/     3000 samples, mode=DIS+VIEW, n=0
            ...
        stage3_ALL_short/    5000 samples, mode=DIS+VIEW+FUTURE(n=1-2)
            ...
        stage4_FUTURE/       10000 samples, mode=mostly FUTURE(n=1-20)
            ...
        manifest.json        metadata
"""

import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import sys, argparse, json, time, contextlib, io
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@contextlib.contextmanager
def suppress_stdout():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


# =====================================================================
# Stage Definitions
# =====================================================================

STAGES = {
    1: {
        "name": "stage1_DIS",
        "description": "Pure image reconstruction (DIS_MOD only)",
        "samples": 2000,
        "mode_probs": [1.0, 0.0, 0.0],  # [DIS, VIEW, FUTURE]
        "n_range": None,
        "epochs_recommended": 10,
        "train_steps_per_epoch": None,  # auto = samples / batch
    },
    2: {
        "name": "stage2_DIS_VIEW",
        "description": "Image reconstruction + viewpoint change",
        "samples": 3000,
        "mode_probs": [0.5, 0.5, 0.0],
        "n_range": None,
        "epochs_recommended": 8,
    },
    3: {
        "name": "stage3_ALL_short",
        "description": "All modes, short-range future (n=1-2)",
        "samples": 5000,
        "mode_probs": [0.2, 0.4, 0.4],
        "n_range": [1, 2],
        "epochs_recommended": 5,
    },
    4: {
        "name": "stage4_FUTURE",
        "description": "Heavy future prediction (n=1-20, Gaussian)",
        "samples": 10000,
        "mode_probs": [0.1, 0.1, 0.8],
        "n_range": "gaussian",  # special handling
        "n_mu_range": [3, 15],
        "n_sig_range": [1, 8],
        "epochs_recommended": 3,
    },
}


# =====================================================================
# Sample Generation (single sample, for use with multiprocessing)
# =====================================================================

# Global render function (set by worker init)
_render_fn = None
_img_size = 64
_delta_t = 0.02
_stereo_gap = 0.2


def _init_worker(module_name, func_id, img_size, delta_t, stereo_gap):
    """Initialize renderer in each worker process."""
    global _render_fn, _img_size, _delta_t, _stereo_gap
    _img_size = img_size
    _delta_t = delta_t
    _stereo_gap = stereo_gap
    sys.path.insert(0, ".")
    mod = __import__(module_name)
    _render_fn = getattr(mod, f"render_scene_{func_id}")


def _render_one(seed, t, az, el, obs):
    with suppress_stdout():
        imgs = _render_fn(
            seed_setup=seed, times=float(t),
            angles=(float(az), float(el)),
            observer_pos=tuple(float(x) for x in obs),
            image_size=(_img_size, _img_size))
    img = imgs[0] if isinstance(imgs, list) else imgs
    return img.astype(np.float32) / 255.0


def _make_stereo_obs(az, oc):
    az_rad = np.radians(az)
    px, py = -np.sin(az_rad), np.cos(az_rad)
    h = _stereo_gap / 2.0
    L = [oc[0]+px*h, oc[1]+py*h, oc[2]]
    R = [oc[0]-px*h, oc[1]-py*h, oc[2]]
    return L, R


def _has_motion(seed, t1, t2, az, el, obs):
    if abs(t2 - t1) < 1e-6:
        return True
    try:
        i1 = _render_one(seed, t1, az, el, obs)
        i2 = _render_one(seed, t2, az, el, obs)
        return np.abs(i1 - i2).mean() > 0.008
    except:
        return False


def generate_one_sample(args_tuple):
    """
    Generate one training sample. Called by multiprocessing pool.
    args_tuple: (sample_idx, mode, n_steps, random_seed)
    Returns: dict with pics, params, target, mode, n_steps (or None on failure)
    """
    sample_idx, mode, n_steps, rand_seed = args_tuple
    rng = np.random.RandomState(rand_seed)

    for _ in range(10):
        seed = rng.randint(0, 100000)
        az, el = rng.uniform(0, 360), rng.uniform(-10, 30)
        oz = rng.uniform(1.5, 3.0)
        oc = [rng.uniform(-2, 2), rng.uniform(-2, 2), oz]
        t1 = rng.uniform(0, 5.0)

        if mode == 'DIS_MOD':
            t2 = t1
            tt, ta, te, to_ = t1, az, el, list(oc)
        elif mode == 'VIEW_MOD':
            t2 = t1 + rng.choice([0, 1]) * _delta_t
            tt = t1
            ta = az + rng.uniform(-15, 15)
            te = el + rng.uniform(-5, 5)
            to_ = [oc[0]+rng.uniform(-1, 1), oc[1]+rng.uniform(-1, 1), oz]
        else:  # FUTURE_MOD
            t2 = t1 + rng.choice([0, 1]) * _delta_t
            tt = t1 + n_steps * _delta_t
            ta = az + rng.uniform(-10, 10)
            te = el + rng.uniform(-3, 3)
            to_ = [oc[0]+rng.uniform(-.5, .5), oc[1]+rng.uniform(-.5, .5), oz]

        # Check motion for FUTURE_MOD
        if mode == 'FUTURE_MOD':
            if not _has_motion(seed, t1, tt, az, el, tuple(oc)):
                continue

        try:
            oL1, oR1 = _make_stereo_obs(az, oc)
            oL2, oR2 = _make_stereo_obs(az, oc)
            pic1 = _render_one(seed, t1, az, el, oL1)
            pic2 = _render_one(seed, t1, az, el, oR1)
            pic3 = _render_one(seed, t2, az, el, oL2)
            pic4 = _render_one(seed, t2, az, el, oR2)
            tgt  = _render_one(seed, tt, ta, te, to_)

            def mkp(t, o, a, e):
                return np.array([t, o[0], o[1], o[2], a, e], np.float32)

            pics = np.stack([pic1, pic2, pic3, pic4]).transpose(0, 3, 1, 2)  # (4,3,H,W)
            params = np.stack([mkp(t1,oL1,az,el), mkp(t1,oR1,az,el),
                               mkp(t2,oL2,az,el), mkp(t2,oR2,az,el),
                               mkp(tt,to_,ta,te)])  # (5,6)
            target = tgt.transpose(2, 0, 1)  # (3,H,W)

            return {
                "pics": pics, "params": params, "target": target,
                "mode": mode, "n_steps": n_steps, "seed": seed,
            }
        except:
            continue

    return None  # failed after retries


# =====================================================================
# Batch Save/Load
# =====================================================================

def save_batch(samples, path):
    """Save a list of sample dicts as a single .npz file."""
    pics = np.stack([s["pics"] for s in samples])       # (N, 4, 3, H, W)
    params = np.stack([s["params"] for s in samples])   # (N, 5, 6)
    targets = np.stack([s["target"] for s in samples])  # (N, 3, H, W)
    n_steps = np.array([s["n_steps"] for s in samples], dtype=np.int32)
    modes = np.array([s["mode"] for s in samples])

    np.savez_compressed(path,
                        pics=pics, params=params, targets=targets,
                        n_steps=n_steps, modes=modes)


def load_batch(path):
    """Load a .npz batch file. Returns dict of arrays."""
    data = np.load(path, allow_pickle=True)
    return {
        "pics": data["pics"],
        "params": data["params"],
        "targets": data["targets"],
        "n_steps": data["n_steps"],
        "modes": data["modes"],
    }


# =====================================================================
# Main Generation Logic
# =====================================================================

def generate_stage(stage_id, module_name, func_id, output_dir,
                   n_samples, img_size, delta_t, stereo_gap,
                   workers, batch_file_size=500):
    """Generate all samples for one stage."""
    stage = STAGES[stage_id]
    stage_dir = os.path.join(output_dir, stage.get("name", f"stage{stage_id}"))
    os.makedirs(stage_dir, exist_ok=True)

    # Override sample count if specified
    if n_samples is None:
        n_samples = stage["samples"]

    print(f"\n{'='*60}")
    print(f"Stage {stage_id}: {stage['description']}")
    print(f"  Samples: {n_samples}, Workers: {workers}")
    print(f"  Mode probs: {stage['mode_probs']}")
    print(f"{'='*60}")

    # Determine mode and n_steps for each sample
    rng = np.random.RandomState(stage_id * 10000)
    mode_names = ['DIS_MOD', 'VIEW_MOD', 'FUTURE_MOD']
    probs = stage["mode_probs"]

    tasks = []
    for i in range(n_samples):
        mode_idx = rng.choice(3, p=probs)
        mode = mode_names[mode_idx]

        if mode == 'FUTURE_MOD':
            n_range = stage.get("n_range")
            if n_range == "gaussian":
                # Sample from evolving Gaussian
                progress = i / max(n_samples - 1, 1)
                mu_lo, mu_hi = stage.get("n_mu_range", [3, 15])
                sig_lo, sig_hi = stage.get("n_sig_range", [1, 8])
                mu = mu_lo + progress * (mu_hi - mu_lo)
                sig = sig_lo + progress * (sig_hi - sig_lo)
                n = max(1, min(int(round(rng.normal(mu, sig))), 50))
            elif n_range is not None:
                n = rng.choice(n_range)
            else:
                n = 1
        else:
            n = 0

        tasks.append((i, mode, n, rng.randint(0, 2**31)))

    # Generate with multiprocessing
    t0 = time.time()
    init_args = (module_name, func_id, img_size, delta_t, stereo_gap)

    results = []
    n_batches = 0

    if workers <= 1:
        # Single process (for debugging)
        _init_worker(*init_args)
        for i, task in enumerate(tasks):
            result = generate_one_sample(task)
            if result is not None:
                results.append(result)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_samples - i - 1) / max(rate, 0.01)
                print(f"\r  [{i+1}/{n_samples}] "
                      f"{rate:.1f} samples/s  ETA: {eta:.0f}s  "
                      f"({len(results)} good)", end="", flush=True)
            # Save batch when full
            if len(results) >= batch_file_size:
                path = os.path.join(stage_dir, f"batch_{n_batches:03d}.npz")
                save_batch(results, path)
                n_batches += 1
                results = []
    else:
        with Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
            for i, result in enumerate(pool.imap(generate_one_sample, tasks,
                                                  chunksize=max(1, min(10, n_samples // workers)))):
                if result is not None:
                    results.append(result)
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (n_samples - i - 1) / max(rate, 0.01)
                    print(f"\r  [{i+1}/{n_samples}] "
                          f"{rate:.1f} samples/s  ETA: {eta:.0f}s  "
                          f"({len(results)} good)", end="", flush=True)
                if len(results) >= batch_file_size:
                    path = os.path.join(stage_dir, f"batch_{n_batches:03d}.npz")
                    save_batch(results, path)
                    n_batches += 1
                    results = []

    # Save remaining
    if results:
        path = os.path.join(stage_dir, f"batch_{n_batches:03d}.npz")
        save_batch(results, path)
        n_batches += 1

    elapsed = time.time() - t0
    total_good = sum(1 for _ in range(n_batches)) * batch_file_size  # approximate
    print(f"\n  Done: {n_batches} batch files, {elapsed:.1f}s "
          f"({n_samples/elapsed:.1f} samples/s)")

    # Save stage metadata
    meta = {
        "stage": stage_id,
        "name": stage["name"],
        "description": stage["description"],
        "n_samples": n_samples,
        "n_batch_files": n_batches,
        "batch_file_size": batch_file_size,
        "mode_probs": stage["mode_probs"],
        "img_size": img_size,
        "delta_t": delta_t,
        "epochs_recommended": stage.get("epochs_recommended", 5),
        "generation_time_s": round(elapsed, 1),
    }
    with open(os.path.join(stage_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    ap = argparse.ArgumentParser(description="Pre-generate O2T training data")
    ap.add_argument("--module", required=True,
                    help="Sealed render module, e.g. render_scene_a1e")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for data")
    ap.add_argument("--stage", type=int, default=None,
                    help="Generate specific stage only (1-4). Default: all")
    ap.add_argument("--samples_per_stage", type=int, default=None,
                    help="Override sample count for all stages")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel workers (default: cpu_count - 1)")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--delta_t", type=float, default=0.02)
    ap.add_argument("--stereo_gap", type=float, default=0.2)
    ap.add_argument("--batch_file_size", type=int, default=500,
                    help="Samples per .npz file")
    args = ap.parse_args()

    func_id = args.module.replace("render_scene_", "")
    workers = args.workers or max(1, cpu_count() - 1)

    print(f"Module: {args.module}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {workers}")
    print(f"Image size: {args.img_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    stages_to_run = [args.stage] if args.stage else [1, 2, 3, 4]
    all_meta = {}

    for sid in stages_to_run:
        meta = generate_stage(
            stage_id=sid,
            module_name=args.module,
            func_id=func_id,
            output_dir=args.output_dir,
            n_samples=args.samples_per_stage,
            img_size=args.img_size,
            delta_t=args.delta_t,
            stereo_gap=args.stereo_gap,
            workers=workers,
            batch_file_size=args.batch_file_size,
        )
        all_meta[f"stage{sid}"] = meta

    # Save overall manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print("All done!")


if __name__ == "__main__":
    main()
