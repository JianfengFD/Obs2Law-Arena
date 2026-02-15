#!/usr/bin/env python3
"""
generate_sky_training_data.py — Practice Mode
==============================================
Generate training images from the virtual telescope.

Usage:
    python generate_sky_training_data.py \
        --sim_data simulation_data_20.0yrs.txt \
        --star_catalog tycho2_entire_sky.fits \
        --n_times 50 --t_max 365 \
        --lon 86 --lat 0 --phi -90 --theta 5

Requires: numpy, scipy, matplotlib, Pillow, astropy
"""

import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from earth_view_mod import VirtualTelescope


def main():
    ap = argparse.ArgumentParser(
        description="Generate sky observation training data (practice mode)")
    ap.add_argument("--sim_data", type=str, required=True,
                    help="Path to simulation_data_XXyrs.txt")
    ap.add_argument("--star_catalog", type=str,
                    default="tycho2_entire_sky.fits",
                    help="Path to Tycho-2 star catalog FITS file")
    ap.add_argument("--output_dir", default="sky_training_data")
    ap.add_argument("--n_times", type=int, default=20,
                    help="Number of time samples")
    ap.add_argument("--t_start", type=float, default=0.0,
                    help="Start time (days)")
    ap.add_argument("--t_max", type=float, default=365.0,
                    help="End time (days)")
    ap.add_argument("--lon", type=float, default=86.0,
                    help="Observer longitude (deg)")
    ap.add_argument("--lat", type=float, default=0.0,
                    help="Observer latitude (deg)")
    ap.add_argument("--phi", type=float, default=-90.0,
                    help="Telescope azimuth (deg)")
    ap.add_argument("--theta", type=float, default=5.0,
                    help="Telescope elevation (deg)")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--tilt", type=float, default=23.439281,
                    help="Earth axial tilt (deg)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize telescope
    telescope = VirtualTelescope(
        simulation_data_path=args.sim_data,
        star_catalog_path=args.star_catalog,
    )

    times = np.linspace(args.t_start, args.t_max, args.n_times)

    # Save metadata
    meta = {
        "sim_data": args.sim_data,
        "observer_lon_deg": args.lon,
        "observer_lat_deg": args.lat,
        "telescope_phi_deg": args.phi,
        "telescope_theta_deg": args.theta,
        "zoom": args.zoom,
        "tilt_deg": args.tilt,
        "times_days": times.tolist(),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Generating {len(times)} sky images...")
    t0 = time.time()

    for i, t_day in enumerate(times):
        print(f"  [{i+1}/{len(times)}] t = {t_day:.2f} days", end=" ... ", flush=True)
        img = telescope.capture(
            time_days=t_day,
            lon_deg=args.lon,
            lat_deg=args.lat,
            phi_deg=args.phi,
            theta_deg=args.theta,
            zoom=args.zoom,
            tilt_deg=args.tilt,
        )
        fname = f"sky_t{t_day:08.2f}.png"
        img.save(os.path.join(args.output_dir, fname))
        print("done")

    total = time.time() - t0
    print(f"\nDone! {len(times)} images in {total:.1f}s")
    print(f"Output: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
