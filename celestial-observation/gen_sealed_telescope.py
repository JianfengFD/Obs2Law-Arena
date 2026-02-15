#!/usr/bin/env python3
"""
gen_sealed_telescope.py — Sealed Telescope Generator
=====================================================
Encrypts orbital trajectory data and generates a sealed VirtualTelescope
module where the Earth's axial tilt and trajectory data are hidden.

Security model:
  - Trajectory data: encrypted with Fernet (AES-128-CBC), key derived from
    a passphrase that only the organiser knows. The encrypted file is
    distributed to participants but is useless without the key.
  - The sealed module embeds the decryption key + tilt_deg inside compiled
    bytecode (same technique as gen_sealed_render.py for the room physics).
  - Participants call capture_XXX(...) to get images, but cannot print()
    trajectories or tilt_deg.

Usage:
    python gen_sealed_telescope.py \
        --sim_data simulation_data_20.0yrs.txt \
        --alpha 0.01 \
        --tilt 23.439281 \
        --func_id T01

Outputs:
    trajectory_T01.enc             — encrypted trajectory data (give to participants)
    sealed_telescope_T01.py        — sealed module (give to participants)
    key_telescope_T01.txt          — answer key (organiser keeps)

Dependencies: cryptography (pip install cryptography)
"""

import os, sys, argparse, hashlib, time, textwrap
import base64, zlib, marshal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_sealed_telescope(sim_data_path: str, alpha: float, tilt_deg: float,
                               func_id: str, output_dir: str = "."):
    # ── 1. Read and encrypt trajectory data ──
    with open(sim_data_path, "r") as f:
        raw_data = f.read()

    # Derive encryption key from a unique passphrase
    passphrase = f"sealed_tele_{func_id}_{alpha}_{tilt_deg}_{hashlib.sha256(raw_data[:200].encode()).hexdigest()[:8]}"
    
    try:
        from cryptography.fernet import Fernet
        # Derive a Fernet key from passphrase
        key_bytes = hashlib.sha256(passphrase.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        cipher = Fernet(fernet_key)
        encrypted_data = cipher.encrypt(raw_data.encode())
        use_fernet = True
    except ImportError:
        # Fallback: simple XOR + base64 (less secure but no extra deps)
        print("Warning: 'cryptography' package not found, using XOR fallback.")
        key_stream = hashlib.sha256(passphrase.encode()).digest()
        data_bytes = raw_data.encode()
        xored = bytes(b ^ key_stream[i % len(key_stream)] for i, b in enumerate(data_bytes))
        encrypted_data = base64.b64encode(zlib.compress(xored, 9))
        use_fernet = False

    enc_path = os.path.join(output_dir, f"trajectory_{func_id}.enc")
    with open(enc_path, "wb") as f:
        f.write(encrypted_data)

    # ── 2. Read earth_view_mod.py source ──
    view_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "earth_view_mod.py")
    with open(view_path, "r") as f:
        view_source = f.read()

    # ── 3. Build wrapper source ──
    # This contains: decryption key, tilt_deg, trajectory data path,
    # and the full VirtualTelescope class — all as local variables
    # inside a closure, compiled to bytecode.

    wrapper_source = f'''
import sys, types, os, hashlib, base64, zlib, tempfile
import numpy as np

_VIEW_SRC = {repr(view_source)}
_PASSPHRASE = {repr(passphrase)}
_USE_FERNET = {use_fernet}
_TILT = {tilt_deg}
_ALPHA = {alpha}

def _build(enc_path, star_catalog_path):
    # Decrypt trajectory data
    with open(enc_path, "rb") as f:
        enc_data = f.read()

    if _USE_FERNET:
        from cryptography.fernet import Fernet
        key_bytes = hashlib.sha256(_PASSPHRASE.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        cipher = Fernet(fernet_key)
        raw_data = cipher.decrypt(enc_data).decode()
    else:
        key_stream = hashlib.sha256(_PASSPHRASE.encode()).digest()
        compressed = base64.b64decode(enc_data)
        xored = zlib.decompress(compressed)
        raw_data = bytes(b ^ key_stream[i % len(key_stream)] for i, b in enumerate(xored)).decode()

    # Write decrypted data to temp file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.write(raw_data)
    tmp.close()
    tmp_path = tmp.name

    # Load VirtualTelescope module
    mod = types.ModuleType("_vt")
    mod.__dict__["__builtins__"] = __builtins__
    sys.modules["_vt"] = mod
    exec(compile(_VIEW_SRC, "<vt>", "exec"), mod.__dict__)

    # Create telescope instance
    telescope = mod.VirtualTelescope(
        simulation_data_path=tmp_path,
        star_catalog_path=star_catalog_path,
    )

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except:
        pass

    _tilt = _TILT
    _alpha_val = _ALPHA

    def _capture(time_days, lon_deg, lat_deg, phi_deg, theta_deg, zoom=1.0):
        return telescope.capture(
            time_days=time_days,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            phi_deg=phi_deg,
            theta_deg=theta_deg,
            zoom=zoom,
            tilt_deg=_tilt,
        )

    def _info():
        print(f"Tracked bodies: {{telescope.tracked_bodies}}")
        print(f"Time range: {{telescope.t_days[0]:.1f}} to {{telescope.t_days[-1]:.1f}} days")
        print(f"Data points: {{len(telescope.t_days)}}")

    return _capture, _info

# Clean up module-level secrets after _build is called
# (they only survive in the closure)
'''

    # ── 4. Compile wrapper to bytecode blob ──
    code_obj = compile(wrapper_source, "<sealed_tele>", "exec")
    blob = base64.b64encode(
        zlib.compress(marshal.dumps(code_obj), 9)).decode()

    checksum = hashlib.sha256(
        f"{func_id}:{alpha}:{tilt_deg}:{time.time()}".encode()
    ).hexdigest()[:12]

    # ── 5. Generate the sealed .py module ──
    sealed_py = textwrap.dedent(f'''\
#!/usr/bin/env python3
"""
sealed_telescope_{func_id} — Sealed Virtual Telescope (Competition)
=====================================================================
Observe the sky from Earth. The orbital dynamics and Earth's axial tilt
are hidden inside compiled bytecode. Your task: discover the underlying
physics from sky observations.

API
---
    from sealed_telescope_{func_id} import capture_{func_id}, info_{func_id}

    # Single image
    img = capture_{func_id}(
        time_days=10.0,     # simulation time in days
        lon_deg=86.0,       # observer longitude (deg)
        lat_deg=0.0,        # observer latitude (deg)
        phi_deg=-90.0,      # telescope azimuth (deg)
        theta_deg=5.0,      # telescope elevation (deg)
        zoom=1.0,           # zoom factor
    )
    img.save("sky.png")     # PIL Image

    # Multiple images (pass lists)
    imgs = capture_{func_id}(
        time_days=[10, 20, 30],
        lon_deg=86.0, lat_deg=0.0,
        phi_deg=-90.0, theta_deg=5.0,
    )

    # Show available info
    info_{func_id}()

Checksum: {checksum}
Requires: trajectory_{func_id}.enc in the working directory (or specify path).
"""

import sys as _sys
import types as _types
import marshal as _marshal
import base64 as _b64
import zlib as _zlib

_BLOB = (
    "{blob}"
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
                                "trajectory_{func_id}.enc")
    # Load bytecode
    raw = _zlib.decompress(_b64.b64decode(_BLOB))
    code = _marshal.loads(raw)
    ns = {{"__builtins__": __builtins__}}
    exec(code, ns)
    # Build telescope
    _capture_fn, _info_fn = ns["_build"](enc_path, star_catalog_path)
    # Clean secrets from namespace
    for k in list(ns.keys()):
        if k.startswith("_") and k not in ("_build", "__builtins__"):
            del ns[k]
    del ns["_build"]
    _initialized = True

def capture_{func_id}(time_days, lon_deg, lat_deg, phi_deg, theta_deg,
                      zoom=1.0, enc_path=None, star_catalog_path="tycho2_entire_sky.fits"):
    """Capture sky image(s). Returns PIL Image or list of PIL Images."""
    _ensure_init(enc_path, star_catalog_path)
    return _capture_fn(time_days, lon_deg, lat_deg, phi_deg, theta_deg, zoom)

def info_{func_id}(enc_path=None, star_catalog_path="tycho2_entire_sky.fits"):
    """Print basic info (tracked bodies, time range). Does NOT reveal tilt or alpha."""
    _ensure_init(enc_path, star_catalog_path)
    _info_fn()

if __name__ == "__main__":
    print("Sealed Virtual Telescope: capture_{func_id}")
    print(f"Checksum: {checksum}")
    print("\\nRun info_{func_id}() after importing to see available data range.")
''')

    sealed_path = os.path.join(output_dir, f"sealed_telescope_{func_id}.py")
    with open(sealed_path, "w") as f:
        f.write(sealed_py)

    # ── 6. Key file ──
    key_path = os.path.join(output_dir, f"key_telescope_{func_id}.txt")
    with open(key_path, "w") as f:
        f.write(f"Competition Answer Key — Celestial Observation\n")
        f.write(f"{'='*55}\n")
        f.write(f"Function ID     : {func_id}\n")
        f.write(f"Checksum        : {checksum}\n")
        f.write(f"Alpha           : {alpha}\n")
        f.write(f"Tilt (deg)      : {tilt_deg}\n")
        f.write(f"Gravity law     : F ∝ (1+alpha) / r^(2+alpha)\n")
        f.write(f"                  alpha = {alpha}\n")
        f.write(f"Sim data source : {sim_data_path}\n")
        f.write(f"Encryption      : {'Fernet (AES)' if use_fernet else 'XOR fallback'}\n")
        f.write(f"{'='*55}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Generated:")
    print(f"  {enc_path}             (encrypted data → participants)")
    print(f"  {sealed_path}  (sealed module → participants)")
    print(f"  {key_path}    (answer key → organiser)")
    print(f"  Alpha={alpha}, Tilt={tilt_deg} deg")

    return enc_path, sealed_path, key_path


def main():
    ap = argparse.ArgumentParser(
        description="Generate sealed telescope module for competition")
    ap.add_argument("--sim_data", type=str, required=True,
                    help="Path to simulation_data_XXyrs.txt")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="Gravity modification alpha (hidden)")
    ap.add_argument("--tilt", type=float, default=23.439281,
                    help="Earth axial tilt in degrees (hidden)")
    ap.add_argument("--func_id", type=str, default=None,
                    help="Function ID suffix (default: auto)")
    ap.add_argument("--output_dir", type=str, default=".")
    args = ap.parse_args()

    if args.func_id is None:
        h = hashlib.md5(f"{args.alpha}_{args.tilt}_{time.time()}".encode())
        args.func_id = "T" + h.hexdigest()[:2]

    os.makedirs(args.output_dir, exist_ok=True)
    generate_sealed_telescope(args.sim_data, args.alpha, args.tilt,
                               args.func_id, args.output_dir)


if __name__ == "__main__":
    main()
