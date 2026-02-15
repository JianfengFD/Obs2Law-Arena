#!/usr/bin/env python3
"""
generate_training_data.py
=========================
Example: how to use physics_sim.py to generate training images for
a neural network that learns to predict future frames / recover
the underlying gravity law.

Usage:
    # Generate 20 scenes, 11 time-steps each, gravity seed = 42
    python generate_training_data.py --n_scenes 20 --seed_g 42

    # Custom observer, more time steps
    python generate_training_data.py --n_scenes 50 --seed_g 7 \
        --obs 0 -5 1.5 --az 30 --el -3 --dt 0.25 --t_max 5

    # Higher resolution
    python generate_training_data.py --n_scenes 10 --image_size 256
"""

import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics_sim import (
    render_scene, _generate_scene, _generate_gravity, get_ball_positions,
)


def main():
    ap = argparse.ArgumentParser(
        description="Generate training data from physics_sim")
    ap.add_argument("--output_dir", default="training_data")
    ap.add_argument("--seed_g", type=int, default=42,
                    help="Gravity seed (defines the physics law)")
    ap.add_argument("--n_scenes", type=int, default=10,
                    help="Number of different scenes to generate")
    ap.add_argument("--seed_start", type=int, default=0,
                    help="First scene seed (scenes use seed_start..seed_start+n)")
    ap.add_argument("--t_max", type=float, default=5.0,
                    help="Maximum simulation time (s)")
    ap.add_argument("--dt", type=float, default=0.5,
                    help="Time step between frames (s)")
    ap.add_argument("--obs", type=float, nargs=3, default=[0, 0, 2.0],
                    metavar=("X", "Y", "Z"),
                    help="Observer position")
    ap.add_argument("--az", type=float, default=25.0,
                    help="Azimuth angle (deg, 0=+X)")
    ap.add_argument("--el", type=float, default=-5.0,
                    help="Elevation angle (deg, 0=horizontal)")
    ap.add_argument("--image_size", type=int, default=512,
                    help="Image size (square, pixels)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    times = np.arange(0, args.t_max + 1e-9, args.dt).tolist()
    obs = tuple(args.obs)
    angle = (args.az, args.el)
    img_size = (args.image_size, args.image_size)

    # ── Gravity ground truth (known to the practitioner) ──
    grav = _generate_gravity(args.seed_g)
    print(f"Gravity law:  g(z) = {grav.g0:.4f} + {grav.alpha:.4f} / (z + 10)")
    print(f"  seed_g = {args.seed_g}")
    print(f"  g(0)   = {grav.g(0):.4f} m/s²")
    print(f"  g(5)   = {grav.g(5):.4f} m/s²")
    print(f"Observer: {obs}   angle: az={args.az}° el={args.el}°")
    print(f"Times: {len(times)} frames, dt={args.dt}s, t_max={args.t_max}s")
    print(f"Image: {img_size[0]}×{img_size[1]}")
    print(f"Scenes: {args.n_scenes} (seeds {args.seed_start}..{args.seed_start + args.n_scenes - 1})")
    print()

    # ── Dataset-level metadata ──
    dataset_meta = {
        "description": "Physics simulation training data",
        "gravity_seed": args.seed_g,
        "gravity_g0": float(grav.g0),
        "gravity_alpha": float(grav.alpha),
        "gravity_formula": f"g(z) = {grav.g0:.6f} + {grav.alpha:.6f} / (z + 10)",
        "observer_pos": list(obs),
        "azimuth_deg": args.az,
        "elevation_deg": args.el,
        "image_size": list(img_size),
        "times": times,
        "n_scenes": args.n_scenes,
        "seed_start": args.seed_start,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(dataset_meta, f, indent=2)

    t0_all = time.time()

    for i in range(args.n_scenes):
        seed_setup = args.seed_start + i
        scene_dir = os.path.join(args.output_dir, f"scene_{seed_setup:04d}")
        os.makedirs(scene_dir, exist_ok=True)

        print(f"── Scene {i+1}/{args.n_scenes}  seed_setup={seed_setup} ──")

        # ── Render all frames ──
        t0 = time.time()
        images = render_scene(
            seed_setup=seed_setup,
            seed_g=args.seed_g,
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
                fname = f"t_{ti:.2f}.png"
                Image.fromarray(img).save(os.path.join(scene_dir, fname))
        except ImportError:
            import imageio
            for ti, img in zip(times, images):
                fname = f"t_{ti:.2f}.png"
                imageio.imwrite(os.path.join(scene_dir, fname), img)

        # ── Scene ground truth (for validation) ──
        scene = _generate_scene(seed_setup)
        scene_meta = {
            "seed_setup": seed_setup,
            "n_balls": len(scene["balls"]),
            "n_ramps": len(scene["ramps"]),
            "balls": [],
            "ramps": [],
        }
        for b in scene["balls"]:
            scene_meta["balls"].append({
                "side": "X>0" if b.side == 1 else "X<0",
                "motion_type": b.motion_type,
                "radius": float(b.radius),
                "pos0": b.pos0.tolist(),
                "vel0": b.vel0.tolist(),
                "color": b.color.tolist(),
            })
        for rm in scene["ramps"]:
            scene_meta["ramps"].append({
                "slope_deg": float(rm.angle_deg),
                "orientation_deg": float(rm.orientation_deg),
                "height": float(rm.height),
                "width": float(rm.width),
                "base_x": float(rm.base_x),
                "base_y": float(rm.base_y),
                "run_length": float(rm.run_length),
            })

        # ── Ball positions at each time (ground truth trajectory) ──
        trajectories = {}
        for ti in times:
            positions = get_ball_positions(seed_setup, args.seed_g, ti)
            trajectories[f"{ti:.2f}"] = [
                p.tolist() if p is not None else None for p in positions
            ]
        scene_meta["trajectories"] = trajectories

        with open(os.path.join(scene_dir, "params.json"), "w") as f:
            json.dump(scene_meta, f, indent=2)

        print(f"  → {len(images)} images saved ({dt_render:.1f}s)")

    total = time.time() - t0_all
    n_imgs = args.n_scenes * len(times)
    print(f"\n{'='*60}")
    print(f"Done! {n_imgs} images in {total:.1f}s "
          f"({total/n_imgs:.2f}s/image)")
    print(f"Output: {os.path.abspath(args.output_dir)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
