#!/usr/bin/env python3
"""
make_animation.py  —  Generate physics simulation → MP4 + GIF
=============================================================
Usage:
    python make_animation.py
    python make_animation.py --seed_setup 42 --seed_g 7 --duration 8
    python make_animation.py --rotate
"""

import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics_sim import render_scene, describe_scene, HAS_PYVISTA


def save_gif(images, path, fps):
    ms = max(1, int(1000.0 / fps))
    try:
        from PIL import Image
        frames = [Image.fromarray(img) for img in images]
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=ms, loop=0, optimize=False)
        print(f"  ✓ GIF: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
        return True
    except Exception as e1:
        try:
            import imageio
            imageio.mimsave(path, images, duration=ms/1000.0, loop=0)
            print(f"  ✓ GIF (imageio): {path}")
            return True
        except Exception as e2:
            print(f"  ✗ GIF failed: {e1} / {e2}")
            return False


def save_mp4(images, path, fps):
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec='libx264')
        for img in images:
            writer.append_data(img)
        writer.close()
        print(f"  ✓ MP4: {path}")
        return True
    except Exception as e:
        print(f"  ⚠ MP4 failed: {e}")
        return False


def save_pngs(images, out_dir="frames"):
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        try:
            from PIL import Image
            Image.fromarray(img).save(f"{out_dir}/frame_{i:04d}.png")
        except ImportError:
            import imageio
            imageio.imwrite(f"{out_dir}/frame_{i:04d}.png", img)
    print(f"  ✓ {len(images)} PNGs → {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_setup", type=int,   default=42)
    ap.add_argument("--seed_g",     type=int,   default=7)
    ap.add_argument("--duration",   type=float, default=6.0)
    ap.add_argument("--fps",        type=int,   default=20)
    ap.add_argument("--width",      type=int,   default=1280)
    ap.add_argument("--height",     type=int,   default=720)
    # ── Camera defaults: look slightly down toward ground ──
    ap.add_argument("--azimuth",    type=float, default=25.0,
                    help="deg, 0=+X")
    ap.add_argument("--elevation",  type=float, default=-5.0,
                    help="deg, negative=look down (default -5)")
    ap.add_argument("--rotate",     action="store_true")
    ap.add_argument("--rotate_speed", type=float, default=15.0)
    ap.add_argument("--observer",   type=float, nargs=3,
                    default=[0.0, 0.0, 1.2])
    ap.add_argument("--output",     type=str,   default="physics_anim.mp4")
    ap.add_argument("--no-gif",     action="store_true")
    args = ap.parse_args()

    if not HAS_PYVISTA:
        print("ERROR: pip install pyvista"); sys.exit(1)

    describe_scene(args.seed_setup, args.seed_g)

    n = int(args.duration * args.fps)
    times = np.linspace(0, args.duration, n, endpoint=False).tolist()
    print(f"Rendering {n} frames ({args.fps}fps × {args.duration:.1f}s) "
          f"{args.width}×{args.height}")

    if args.rotate:
        az_arr = args.azimuth + args.rotate_speed * np.array(times)
        angles = [(az, args.elevation) for az in az_arr]
    else:
        angles = (args.azimuth, args.elevation)

    t0 = time.time()
    images = render_scene(args.seed_setup, args.seed_g, times, angles,
                          tuple(args.observer), (args.width, args.height))
    dt = time.time() - t0
    print(f"\n  {len(images)} frames in {dt:.1f}s ({dt/len(images):.2f}s/f)\n")

    # ── Outputs ──
    if not save_mp4(images, args.output, args.fps):
        save_pngs(images)

    if not args.no_gif:
        save_gif(images, os.path.splitext(args.output)[0] + ".gif", args.fps)

    # previews
    for tag, idx in [("first", 0), ("last", -1)]:
        try:
            from PIL import Image
            Image.fromarray(images[idx]).save(f"frame_{tag}.png")
        except Exception:
            pass
    print("Done! ✓")


if __name__ == "__main__":
    main()
