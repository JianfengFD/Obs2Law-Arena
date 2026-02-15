#!/usr/bin/env python3
"""
gen_sealed_render.py — Sealed Renderer Generator (v3, secure)
==============================================================
Security: seed_g exists ONLY inside compiled bytecode (marshal).
The output .py file contains one base64 blob + a thin loader.
To extract seed_g, one must reverse-engineer CPython bytecode.

Usage:
    python gen_sealed_render.py --seed_g 42
    python gen_sealed_render.py --seed_g 42 --func_id a1b
"""

import os, sys, argparse, textwrap, hashlib, time
import base64, zlib, marshal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_sealed(seed_g: int, func_id: str, output_dir: str = "."):
    # ── Read physics_sim.py source ──
    sim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "physics_sim.py")
    with open(sim_path, "r") as f:
        sim_source = f.read()

    # ── Build a SINGLE wrapper source that includes everything ──
    # seed_g is a literal integer inside this source.
    # This entire source gets compiled → marshalled → compressed → base64.
    # The output .py never contains this source as text.

    wrapper_source = f'''
import sys, types, numpy as np

_SRC = {repr(sim_source)}

def _build():
    # Load engine
    mod = types.ModuleType("_pe")
    mod.__dict__["__builtins__"] = __builtins__
    sys.modules["_pe"] = mod
    exec(compile(_SRC, "<pe>", "exec"), mod.__dict__)

    _g = {seed_g}  # local variable, captured by closures only

    def _render(seed_setup, times, angles,
                observer_pos=(0.0, 0.0, 2.0), image_size=(512, 512)):
        return mod.render_scene(seed_setup, _g, times, angles,
                                observer_pos, image_size)

    def _describe(seed_setup):
        scene = mod._generate_scene(seed_setup)
        print(f"Scene seed_setup={{seed_setup}}")
        print(f"  Balls: {{len(scene['balls'])}}  Ramps: {{len(scene['ramps'])}}")
        for i, b in enumerate(scene["balls"]):
            tag = "X>0" if b.side == 1 else "X<0"
            print(f"  Ball {{i}} [{{tag}}] {{b.motion_type}} r={{b.radius:.3f}}m")
            print(f"    pos0={{np.round(b.pos0,3)}}  vel0={{np.round(b.vel0,3)}}")
        for i, rm in enumerate(scene["ramps"]):
            print(f"  Ramp {{i}}: slope={{rm.angle_deg:.1f}} deg "
                  f"orient={{rm.orientation_deg:.1f}} deg")
            print(f"    h={{rm.height:.2f}}m w={{rm.width:.2f}}m "
                  f"base=({{rm.base_x:.2f}},{{rm.base_y:.2f}})")

    return _render, _describe, mod

render, describe, _engine_ref = _build()
del _build, _SRC  # clean up
'''

    # ── Compile → marshal → compress → base64 ──
    code_obj = compile(wrapper_source, "<sealed>", "exec")
    blob = base64.b64encode(
        zlib.compress(marshal.dumps(code_obj), 9)).decode()

    checksum = hashlib.sha256(
        f"{func_id}:{seed_g}:{time.time()}".encode()).hexdigest()[:12]

    from physics_sim import _generate_gravity
    grav = _generate_gravity(seed_g)

    # ── Generate the output .py ──
    # This file is READABLE but contains NO secrets.
    # It only has: docstring + base64 blob + thin loader + public API.
    output_py = textwrap.dedent(f'''\
#!/usr/bin/env python3
"""
render_scene_{func_id} — Sealed Physics Renderer (Competition)
================================================================
The gravity law is hidden inside compiled bytecode.
Your task: discover g(z) by observing the physics.

API
---
    from render_scene_{func_id} import render_scene_{func_id}

    images = render_scene_{func_id}(
        seed_setup=42,
        times=[0, 0.5, 1.0, 2.0],
        angles=(25, -5),           # (azimuth_deg, elevation_deg)
        observer_pos=(0, 0, 2.0),  # camera position
        image_size=(512, 512),     # square output
    )
    # Returns: list of HxWx3 uint8 numpy arrays (RGB)

    describe_scene_{func_id}(seed_setup=42)
    # Prints scene layout (does NOT reveal gravity)

Checksum: {checksum}
"""

import sys as _sys
import types as _types
import marshal as _marshal
import base64 as _b64
import zlib as _zlib

# Sealed bytecode — contains physics engine + hidden parameters
_BLOB = (
    "{blob}"
)

def _boot():
    raw = _zlib.decompress(_b64.b64decode(_BLOB))
    code = _marshal.loads(raw)
    ns = {{"__builtins__": __builtins__}}
    exec(code, ns)
    return ns

_W = _boot()

def render_scene_{func_id}(seed_setup, times, angles,
                           observer_pos=(0.0, 0.0, 2.0),
                           image_size=(512, 512)):
    """Render physics scene. Returns list of HxWx3 uint8 arrays."""
    return _W["render"](seed_setup, times, angles,
                        observer_pos, image_size)

def describe_scene_{func_id}(seed_setup):
    """Print scene info (ball sizes, positions, ramp — NO gravity)."""
    _W["describe"](seed_setup)

# For interactive viewer compatibility
_engine = _W.get("_engine_ref") if "_engine_ref" in _W else None

if __name__ == "__main__":
    print("Sealed render module: render_scene_{func_id}")
    print(f"Checksum: {checksum}")
    print()
    describe_scene_{func_id}(0)
''')

    # ── Write files ──
    sealed_path = os.path.join(output_dir, f"render_scene_{func_id}.py")
    with open(sealed_path, "w") as f:
        f.write(output_py)

    key_path = os.path.join(output_dir, f"key_{func_id}.txt")
    with open(key_path, "w") as f:
        f.write(f"Competition Answer Key\n")
        f.write(f"{'='*50}\n")
        f.write(f"Function ID : {func_id}\n")
        f.write(f"Checksum    : {checksum}\n")
        f.write(f"seed_g      : {seed_g}\n")
        f.write(f"Gravity     : g(z) = {grav.g0:.6f} + {grav.alpha:.6f} / (z + 10)\n")
        f.write(f"  g(0)  = {grav.g(0):.6f} m/s^2\n")
        f.write(f"  g(5)  = {grav.g(5):.6f} m/s^2\n")
        f.write(f"  g(10) = {grav.g(10):.6f} m/s^2\n")
        f.write(f"{'='*50}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Generated:")
    print(f"  {sealed_path}  (participants)")
    print(f"  {key_path}        (secret)")
    print(f"  Gravity: g(z) = {grav.g0:.4f} + {grav.alpha:.4f}/(z+10)")
    return sealed_path, key_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_g", type=int, required=True)
    ap.add_argument("--func_id", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=".")
    args = ap.parse_args()
    if args.func_id is None:
        h = hashlib.md5(f"{args.seed_g}_{time.time()}".encode())
        args.func_id = h.hexdigest()[:3]
    os.makedirs(args.output_dir, exist_ok=True)
    generate_sealed(args.seed_g, args.func_id, args.output_dir)

if __name__ == "__main__":
    main()
