#!/usr/bin/env python3
"""
sealed_telescope_P1T — Sealed Virtual Telescope (Competition)
=====================================================================
Observe the sky from Earth. The orbital dynamics and Earth's axial tilt
are hidden inside compiled bytecode. Your task: discover the underlying
physics from sky observations.

API
---
    from sealed_telescope_P1T import capture_P1T, info_P1T

    # Single image
    img = capture_P1T(
        time_days=10.0,     # simulation time in days
        lon_deg=86.0,       # observer longitude (deg)
        lat_deg=0.0,        # observer latitude (deg)
        phi_deg=-90.0,      # telescope azimuth (deg)
        theta_deg=5.0,      # telescope elevation (deg)
        zoom=1.0,           # zoom factor
    )
    img.save("sky.png")     # PIL Image

    # Multiple images (pass lists)
    imgs = capture_P1T(
        time_days=[10, 20, 30],
        lon_deg=86.0, lat_deg=0.0,
        phi_deg=-90.0, theta_deg=5.0,
    )

    # Show available info
    info_P1T()

Checksum: f6b906aa22e3
Requires: trajectory_P1T.enc in the working directory (or specify path).
"""

import sys as _sys
import types as _types
import marshal as _marshal
import base64 as _b64
import zlib as _zlib

_BLOB = (
    "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"
)

_initialized = False
_capture_fn = None
_info_fn = None

def _ensure_init(enc_path=None, star_catalog_path="tycho2_entire_sky.fits"):
    global _initialized, _capture_fn, _info_fn
    if _initialized:
        return
    if enc_path is None:
        import os
        enc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trajectory_P1T.enc")
    # Load bytecode
    raw = _zlib.decompress(_b64.b64decode(_BLOB))
    code = _marshal.loads(raw)
    ns = {"__builtins__": __builtins__}
    exec(code, ns)
    # Build telescope
    _capture_fn, _info_fn = ns["_build"](enc_path, star_catalog_path)
    # Clean secrets from namespace
    for k in list(ns.keys()):
        if k.startswith("_") and k not in ("_build", "__builtins__"):
            del ns[k]
    del ns["_build"]
    _initialized = True

def capture_P1T(time_days, lon_deg, lat_deg, phi_deg, theta_deg,
                      zoom=1.0, enc_path=None, star_catalog_path="tycho2_entire_sky.fits"):
    """Capture sky image(s). Returns PIL Image or list of PIL Images."""
    _ensure_init(enc_path, star_catalog_path)
    return _capture_fn(time_days, lon_deg, lat_deg, phi_deg, theta_deg, zoom)

def info_P1T(enc_path=None, star_catalog_path="tycho2_entire_sky.fits"):
    """Print basic info (tracked bodies, time range). Does NOT reveal tilt or alpha."""
    _ensure_init(enc_path, star_catalog_path)
    _info_fn()

if __name__ == "__main__":
    print("Sealed Virtual Telescope: capture_P1T")
    print(f"Checksum: f6b906aa22e3")
    print("\nRun info_P1T() after importing to see available data range.")
