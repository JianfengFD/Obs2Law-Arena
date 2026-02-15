#!/usr/bin/env python3
"""
generate_sky_training_data_sealed.py — Competition Mode
========================================================
Generate training images using a sealed telescope module.
Participants use this script — they cannot access trajectory
data or tilt_deg directly.

Usage:
    python generate_sky_training_data_sealed.py \
        --module sealed_telescope_T01 \
        --n_times 50 --t_max 365 \
        --lon 86 --lat 0 --phi -90 --theta 5
"""

import os, sys, json, argparse, time, importlib
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Generate sky training data from sealed telescope")
    ap.add_argument("--module", type=str, required=True,
                    help="Sealed module name, e.g. sealed_telescope_T01")
    ap.add_argument("--output_dir", default="sky_training_data")
    ap.add_argument("--n_times", type=int, default=20)
    ap.add_argument("--t_start", type=float, default=0.0)
    ap.add_argument("--t_max", type=float, default=365.0)
    ap.add_argument("--lon", type=float, default=86.0,
                    help="Observer longitude (deg)")
    ap.add_argument("--lat", type=float, default=0.0,
                    help="Observer latitude (deg)")
    ap.add_argument("--phi", type=float, default=-90.0,
                    help="Telescope azimuth (deg)")
    ap.add_argument("--theta", type=float, default=5.0,
                    help="Telescope elevation (deg)")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--enc_path", type=str, default=None,
                    help="Path to trajectory_XXX.enc (default: auto-detect)")
    ap.add_argument("--star_catalog", type=str,
                    default="tycho2_entire_sky.fits")
    args = ap.parse_args()

    # Import sealed module
    mod_name = args.module
    func_id = mod_name.replace("sealed_telescope_", "")
    sys.path.insert(0, ".")
    mod = importlib.import_module(mod_name)

    capture_fn = getattr(mod, f"capture_{func_id}")
    info_fn = getattr(mod, f"info_{func_id}", None)

    print(f"Using sealed module: {mod_name}")
    
    # Initialize (triggers decryption + telescope setup)
    kwargs = {}
    if args.enc_path:
        kwargs["enc_path"] = args.enc_path
    kwargs["star_catalog_path"] = args.star_catalog

    if info_fn:
        info_fn(**kwargs)
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    times = np.linspace(args.t_start, args.t_max, args.n_times)

    # Save metadata (no hidden params)
    meta = {
        "sealed_module": mod_name,
        "observer_lon_deg": args.lon,
        "observer_lat_deg": args.lat,
        "telescope_phi_deg": args.phi,
        "telescope_theta_deg": args.theta,
        "zoom": args.zoom,
        "times_days": times.tolist(),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Generating {len(times)} sky images...")
    t0 = time.time()

    for i, t_day in enumerate(times):
        print(f"  [{i+1}/{len(times)}] t = {t_day:.2f} days", end=" ... ", flush=True)
        img = capture_fn(
            time_days=t_day,
            lon_deg=args.lon,
            lat_deg=args.lat,
            phi_deg=args.phi,
            theta_deg=args.theta,
            zoom=args.zoom,
            **kwargs,
        )
        # Handle single or list return
        if isinstance(img, list):
            img = img[0]
        fname = f"sky_t{t_day:08.2f}.png"
        img.save(os.path.join(args.output_dir, fname))
        print("done")

    total = time.time() - t0
    print(f"\nDone! {len(times)} images in {total:.1f}s")
    print(f"Output: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
