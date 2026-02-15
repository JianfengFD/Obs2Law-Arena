# Competition Sealed Renderer — Organiser Guide

## Overview

`gen_sealed_render.py` generates a **sealed black-box renderer** for physics simulation competitions. The gravity law `g(z) = g0 + alpha/(z+10)` is hidden inside the module; participants must discover it by observing the physics through rendered images.

---

## Quick Start

### Generate a sealed module

```bash
python gen_sealed_render.py --seed_g 42 --func_id a1b
```

Output:
- `render_scene_a1b.py` — **give to participants**
- `key_a1b.txt` — **keep secret** (contains seed_g and gravity formula)

### What participants receive

1. `render_scene_a1b.py` — the sealed renderer
2. `generate_training_data_sealed.py` — batch data generation script
3. `interactive_viewer_sealed.py` — optional, for visual exploration

### How participants use it

```python
from render_scene_a1b import render_scene_a1b

# Render frames
images = render_scene_a1b(
    seed_setup=42,              # scene layout
    times=[0, 0.5, 1.0, 2.0],  # time points
    angles=(25, -5),            # (azimuth, elevation) degrees
)
# images: list of 512x512x3 uint8 numpy arrays

# Generate training data in batch
# python generate_training_data_sealed.py --module render_scene_a1b --n_scenes 200
```

---

## Security Analysis

### What is protected (3 layers)

| Layer | Technique | What it blocks |
|---|---|---|
| **Source-level** | Physics engine compiled to bytecode → zlib → base64 | `grep`, reading the .py file, text search for seed_g/gravity constants |
| **Namespace-level** | Closure pattern: `_g` is a local variable inside `_build()`, then `del _build, _SRC` | `_W["_G"]`, `_W["seed_g"]`, `_W["_SRC"]` — all return "not found" |
| **API-level** | `describe_scene_XXX()` only prints ball/ramp geometry, never gravity | Casual API exploration reveals nothing |

### What is NOT fully protected

| Attack | Difficulty | Mitigation |
|---|---|---|
| **Closure inspection**: `render.__closure__[0].cell_contents` returns the integer seed_g | Medium (need to know Python internals) | See "Extra hardening" below |
| **Bytecode disassembly**: `dis.dis()` or `marshal.loads()` on the blob to recover wrapper source | Medium-Hard | Obfuscation layer can be added |
| **Brute-force seed_g**: Try all integers 0-99999 and compare output | Medium (computationally feasible) | Use large offset: `actual_seed = seed_g + SECRET_OFFSET` |

### Recommended: Extra hardening with offset

The simplest and most effective hardening: **use a large secret offset** so that even if someone extracts the integer from the closure, it's meaningless without knowing the offset.

```bash
# In your setup (NOT shared with participants):
SECRET_OFFSET = 8374291  # pick any large number, keep it private

# Generate with offset applied:
python gen_sealed_render.py --seed_g $((42 + 8374291)) --func_id a1b

# In key_a1b.txt you'll see the offset seed_g, but you know:
#   real gravity = _generate_gravity(42 + 8374291)
#   which is the SAME as _generate_gravity(8374333)
```

Now even if a participant extracts `8374333` from the closure, they cannot determine what the "real" seed was — and more importantly, **they cannot tell if it's the actual seed_g or an offset version**, since the gravity function is deterministic from any integer.

For competition purposes, what matters is whether participants can recover `g(z)` — the function itself, not the seed. Even with the seed exposed, they'd still need to know the formula `g(z) = g0 + alpha/(z+10)` to use it.

### Practical security assessment

For a **classroom/workshop** setting: the current protection is **more than sufficient**. Participants would need to deliberately hack the module rather than solve the physics problem.

For a **serious competition with prizes**: add the offset trick above, and consider distributing as `.pyc` files only (compiled bytecode, delete the `.py`).

```bash
# Distribute only .pyc:
python -c "import py_compile; py_compile.compile('render_scene_a1b.py')"
# Give participants __pycache__/render_scene_a1b.cpython-3XX.pyc
# Rename to render_scene_a1b.pyc and delete the .py
```

---

## File Inventory

| File | Who gets it | Purpose |
|---|---|---|
| `gen_sealed_render.py` | Organiser only | Generates sealed modules |
| `physics_sim.py` | Organiser only | Core physics engine (source) |
| `key_XXX.txt` | Organiser only | Answer key with seed_g and g(z) |
| `render_scene_XXX.py` | Participants | Black-box renderer |
| `generate_training_data_sealed.py` | Participants | Batch data generation |
| `interactive_viewer_sealed.py` | Participants | Visual exploration tool |
| `README_PRACTICE.md` | Participants (practice mode) | Full documentation with physics_sim exposed |

---

## Generating Multiple Competition Problems

```bash
# Problem A: one gravity law
python gen_sealed_render.py --seed_g 42 --func_id A01

# Problem B: different gravity law
python gen_sealed_render.py --seed_g 7 --func_id B01

# With offset hardening
python gen_sealed_render.py --seed_g 9999942 --func_id C01
```

Each generates an independent sealed module. Participants can be given different problems or the same one.

---

## Answer Verification

Use `score_room.py` to evaluate participant submissions:

```bash
# Single problem
python score_room.py --submission submit_room.txt --keys key_a1b.txt

# Multiple problems
python score_room.py --submission submit_room.txt --keys key_a1b.txt key_c3f.txt

# Or point to a directory of key files
python score_room.py --submission submit_room.txt --key_dir ./keys/
```

Submission file format (`submit_room.txt`):
```
[a1b]
g0 = 9.78
alpha = -1.2

[c3f]
g0 = 9.5
alpha = 3.0
```

Score formula:
```
score = 0.7 × mean|g_sub(z) - g_true(z)|       (functional match, z ∈ [0,20])
      + 0.3 × (|g0_err|/|g0_true| + |α_err|/max(|α_true|, 0.1))  (parameter match)
```

Weights can be adjusted via `--w_func` and `--w_params`.
