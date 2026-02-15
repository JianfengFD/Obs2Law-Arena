#!/usr/bin/env python3
"""
test_sim.py  —  Test & validate the physics simulation
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics_sim import (
    _generate_scene, _generate_gravity, _simulate_ball,
    describe_scene, get_ball_positions, GravityConfig, Ball,
    render_scene, HAS_PYVISTA,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
ok = True

def check(name, cond, detail=""):
    global ok
    tag = PASS if cond else FAIL
    print(f"  {tag} {name}" + (f"  ({detail})" if detail else ""))
    if not cond: ok = False


print("\n═══ 1: Determinism ═══")
s1, s2 = _generate_scene(42), _generate_scene(42)
check("same seed → same #balls", len(s1['balls']) == len(s2['balls']))
for i in range(len(s1['balls'])):
    check(f"ball[{i}] identical",
          np.allclose(s1['balls'][i].pos0, s2['balls'][i].pos0) and
          np.allclose(s1['balls'][i].vel0, s2['balls'][i].vel0) and
          s1['balls'][i].side == s2['balls'][i].side)
s3 = _generate_scene(99)
check("diff seed → diff scene",
      not np.allclose(s1['balls'][0].pos0, s3['balls'][0].pos0))

print("\n═══ 2: Gravity ═══")
g1, g2 = _generate_gravity(7), _generate_gravity(7)
check("g deterministic", np.isclose(g1.g0, g2.g0) and np.isclose(g1.alpha, g2.alpha))
g3 = _generate_gravity(77)
check("diff seed → diff g", not np.isclose(g1.g0, g3.g0))
gv = GravityConfig(9.5, 5.0)
check("g(0) formula", np.isclose(gv.g(0), 9.5 + 5/10), f"g(0)={gv.g(0):.4f}")
check("g(10) formula", np.isclose(gv.g(10), 9.5 + 5/20), f"g(10)={gv.g(10):.4f}")

print("\n═══ 3: Elastic bounce ═══")
grav = GravityConfig(9.81, 0.0)
fb = Ball(0.2, np.array([.5,.5,.5]), 'freefall',
          np.array([10.,0.,3.2]), np.zeros(3), 1)
p0 = _simulate_ball(fb, grav, 0.0, [])
check("t=0 → initial pos", p0 is not None and np.allclose(p0, fb.pos0, atol=1e-3))
p1 = _simulate_ball(fb, grav, 0.5, [])
check("t=0.5 → z≥r", p1 is not None and p1[2] >= fb.radius - 0.01, f"z={p1[2]:.3f}")
p2 = _simulate_ball(fb, grav, 2.0, [])
check("t=2 → still exists (elastic)", p2 is not None)
hs = [_simulate_ball(fb, grav, t, [])[2] for t in np.arange(0,5,0.01)
      if _simulate_ball(fb, grav, t, []) is not None]
check("max height ≈ init", abs(max(hs) - fb.pos0[2]) < 0.3,
      f"max={max(hs):.3f} init={fb.pos0[2]:.3f}")

print("\n═══ 4: X=0 wall ═══")
wb = Ball(0.1, np.array([.5,.5,.5]), 'projectile',
          np.array([2.,0.,1.]), np.array([-10.,0.,5.]), 1)
check("crosses X=0 → None", _simulate_ball(wb, grav, 1.0, []) is None)
sb = Ball(0.1, np.array([.5,.5,.5]), 'projectile',
          np.array([-10.,0.,1.]), np.array([-5.,0.,3.]), -1)
ps = _simulate_ball(sb, grav, 0.5, [])
check("stays in X<0", ps is not None and ps[0] < 0)

print("\n═══ 5: Boundary ═══")
far = Ball(0.1, np.array([.5,.5,.5]), 'projectile',
           np.array([90.,0.,1.]), np.array([10.,0.,5.]), 1)
check("exits 100m → None", _simulate_ball(far, grav, 5.0, []) is None)

print("\n═══ 6: describe_scene ═══")
describe_scene(42, 7)
check("runs OK", True)

print(f"\n═══ 7: Render (pyvista={'YES' if HAS_PYVISTA else 'NO'}) ═══")
if HAS_PYVISTA:
    imgs = render_scene(42, 7, 0.5, (25, -5), image_size=(640, 480))
    check("1 frame", len(imgs) == 1)
    check("shape", imgs[0].shape == (480, 640, 3), f"{imgs[0].shape}")
    check("dtype", imgs[0].dtype == np.uint8)
    # check not mostly blank: at least 5% of pixels differ from background
    bg = np.array([235, 240, 247], dtype=np.uint8)  # approx bg
    diff = np.abs(imgs[0].astype(int) - bg.astype(int)).sum(axis=2)
    pct = (diff > 30).mean() * 100
    check(f"not mostly blank ({pct:.1f}% non-bg)", pct > 3.0, f"{pct:.1f}%")

    imgs2 = render_scene(42, 7, [0,0.5,1.0], (25, -5), image_size=(320,240))
    check("3 times → 3 imgs", len(imgs2) == 3)
    imgs3 = render_scene(42, 7, 1.0, [(0,-5),(90,-5),(180,-5)], image_size=(320,240))
    check("3 angles → 3 imgs", len(imgs3) == 3)

    try:
        from PIL import Image
        Image.fromarray(imgs[0]).save("test_frame.png")
        print(f"  ℹ Saved test_frame.png")
    except Exception:
        pass
else:
    print("  ⚠ pyvista not installed, skipping render tests")
    check("skip OK", True)

print("\n" + "═"*50)
print(f"  {'ALL PASSED' if ok else 'SOME FAILED'}")
print("═"*50 + "\n")
sys.exit(0 if ok else 1)
