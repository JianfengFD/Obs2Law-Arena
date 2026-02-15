#!/usr/bin/env python3
"""
generate_training_data_sealed.py
================================
Generate training data using a sealed render_scene_XXX module.

Usage:
    python generate_training_data_sealed.py --module render_scene_a1b --n_scenes 20
    python generate_training_data_sealed.py --module render_scene_a1b \
        --n_scenes 100 --dt 0.25 --t_max 8 --obs 0 -2 2.0

This is the script that PARTICIPANTS use — they never see seed_g.
"""

import os, sys, json, argparse, time, importlib
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Generate training data from a sealed renderer")
    ap.add_argument("--module", type=str, required=True,
                    help="Sealed module name, e.g. render_scene_a1b")
    ap.add_argument("--output_dir", default="training_data")
    ap.add_argument("--n_scenes", type=int, default=10)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--t_max", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--obs", type=float, nargs=3, default=[0, 0, 2.0],
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--az", type=float, default=25.0)
    ap.add_argument("--el", type=float, default=-5.0)
    ap.add_argument("--image_size", type=int, default=512)
    args = ap.parse_args()

    # ── Import the sealed module ──
    mod_name = args.module
    func_id = mod_name.replace("render_scene_", "")

    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        # try adding current directory
        sys.path.insert(0, ".")
        mod = importlib.import_module(mod_name)

    render_fn = getattr(mod, f"render_scene_{func_id}")
    describe_fn = getattr(mod, f"describe_scene_{func_id}", None)

    print(f"Using sealed module: {mod_name}")
    print(f"  render function:   render_scene_{func_id}()")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    times = np.arange(0, args.t_max + 1e-9, args.dt).tolist()
    obs = tuple(args.obs)
    angle = (args.az, args.el)
    img_size = (args.image_size, args.image_size)

    print(f"Observer: {obs}   angle: az={args.az} el={args.el}")
    print(f"Times: {len(times)} frames, dt={args.dt}s, t_max={args.t_max}s")
    print(f"Image: {img_size[0]}x{img_size[1]}")
    print(f"Scenes: {args.n_scenes}")
    print()

    # ── Dataset metadata ──
    meta = {
        "sealed_module": mod_name,
        "observer_pos": list(obs),
        "azimuth_deg": args.az,
        "elevation_deg": args.el,
        "image_size": list(img_size),
        "times": times,
        "n_scenes": args.n_scenes,
        "seed_start": args.seed_start,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    t0_all = time.time()

    for i in range(args.n_scenes):
        seed_setup = args.seed_start + i
        scene_dir = os.path.join(args.output_dir, f"scene_{seed_setup:04d}")
        os.makedirs(scene_dir, exist_ok=True)

        print(f"-- Scene {i+1}/{args.n_scenes}  seed_setup={seed_setup} --")
        if describe_fn:
            describe_fn(seed_setup)

        # ── Render ──
        t0 = time.time()
        images = render_fn(
            seed_setup=seed_setup,
            times=times,
            angles=angle,
            observer_pos=obs,
            image_size=img_size,
        )
        dt_render = time.time() - t0

        # ── Save images ──
        try:
            from PIL import Image
            for ti, img in zip(times, images):
                Image.fromarray(img).save(
                    os.path.join(scene_dir, f"t_{ti:.2f}.png"))
        except ImportError:
            import imageio
            for ti, img in zip(times, images):
                imageio.imwrite(
                    os.path.join(scene_dir, f"t_{ti:.2f}.png"), img)

        print(f"  -> {len(images)} images ({dt_render:.1f}s)")

    total = time.time() - t0_all
    n_imgs = args.n_scenes * len(times)
    print(f"\nDone! {n_imgs} images in {total:.1f}s")
    print(f"Output: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
