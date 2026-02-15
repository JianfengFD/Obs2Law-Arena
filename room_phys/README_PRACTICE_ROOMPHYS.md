# Physics Simulation Practice — README

## Project Goal

This project provides a **virtual 3D physics environment** where balls undergo projectile motion, free-fall, or roll down inclined ramps under a configurable gravity law. Given an observer's position and viewing angle, the system renders photo-realistic images of the scene at any point in time.

The purpose is to serve as a **practice dataset generator** for researchers building neural networks that learn physical laws from visual observation. Specifically:

> **Task**: Given a sequence of observed images, predict future frames and/or recover the underlying gravity function g(z).

Because the gravity law is controlled by a single seed (`seed_g`), practitioners have full access to the ground truth during training and validation — making this an ideal testbed before tackling real-world physics inference problems.

---

## How It Works

### The Physics

The gravity in this world is **height-dependent**:

```
g(z) = g0 + alpha / (z + 10)
```

where `g0` and `alpha` are determined by `seed_g`. Different seeds produce different gravity laws. At `z=0` (ground level), gravity is roughly 9–10 m/s²; at higher altitudes it changes smoothly.

Each scene (controlled by `seed_setup`) contains:

- **2 balls** — one on each side of x=0, with Gaussian-distributed radii (mean ≈ 0.4m) and random colors
- **0 or 1 ramp** — an inclined plane with random slope, orientation, and position (X>0 side only)
- **A ground plane** — with subtle grid lines

Ball motion types:
- **Projectile**: launched with an initial velocity vector
- **Free-fall**: dropped from a height with zero initial velocity
- **Ramp**: placed on the ramp surface, starts from rest, rolls down under gravity

### The Observer

Images are rendered from a strict pin-hole camera model:
- Position: default `(0, 0, 2.0)` — roughly human eye height at the origin
- Azimuth and elevation control the viewing direction
- The camera is always level (no head tilt)
- Fixed 60° field of view, no zoom

### Visual Cues in the Image

Each rendered frame contains (no text overlays):
- Balls with shadows projected onto the ground (dark discs at z≈0)
- Red scale bars near the origin: one at `(3, 0, 0.1)` for the X>0 ball, one at `(-3, 0, 0.1)` for the X<0 ball, each with length equal to the ball's diameter
- Ground grid lines and ramp geometry with height tick marks

---

## File Overview

| File | Description |
|---|---|
| `physics_sim.py` | Core engine: scene generation, physics simulation, rendering |
| `generate_training_data.py` | Example script: batch-generate training images + metadata |
| `interactive_viewer.py` | GUI viewer (PyQt5 + PyVista) for exploring scenes interactively |
| `make_animation.py` | Create GIF/MP4 animations of a scene |

---

## Quick Start

### Install Dependencies

```bash
pip install numpy pyvista imageio
# For interactive viewer:
pip install pyvistaqt PyQt5
# For MP4 export (optional):
pip install imageio[ffmpeg]
```

### Generate Training Data

```bash
# 20 scenes, gravity seed 42, default observer at (0,0,2.0)
python generate_training_data.py --n_scenes 20 --seed_g 42

# Custom settings
python generate_training_data.py \
    --n_scenes 100 \
    --seed_g 7 \
    --seed_start 0 \
    --obs 0 -2 1.5 \
    --az 30 --el -3 \
    --dt 0.25 --t_max 8 \
    --image_size 512
```

This produces:

```
training_data/
  metadata.json              ← dataset-level info (gravity law, observer, etc.)
  scene_0000/
    params.json              ← ground truth (ball positions, velocities, ramp geometry)
    t_0.00.png               ← rendered frame at t=0
    t_0.50.png               ← rendered frame at t=0.5s
    ...
  scene_0001/
    ...
```

### Use the API Directly

```python
from physics_sim import render_scene, describe_scene

# Inspect a scene
describe_scene(seed_setup=42, seed_g=7)

# Render one frame
images = render_scene(
    seed_setup=42, seed_g=7,
    times=2.0,
    angles=(25, -5),               # azimuth=25°, elevation=-5°
    observer_pos=(0, 0, 2.0),
)
# images[0] is a 512×512×3 uint8 numpy array

# Render a time sequence
images = render_scene(
    seed_setup=42, seed_g=7,
    times=[0, 0.5, 1.0, 1.5, 2.0],
    angles=(25, -5),
)

# Multiple observer positions (same scene, same time)
images = render_scene(
    seed_setup=42, seed_g=7,
    times=1.0,
    angles=(25, -5),
    observer_pos=[(0,0,2.0), (5,0,1.2), (-5,3,2.0)],
)
```

### Explore Interactively

```bash
python interactive_viewer.py --seed_setup 42 --seed_g 7
```

The GUI shows the 3D scene with play/pause controls, time slider, speed selector, and live camera parameter readout. You can rotate/zoom with the mouse and see azimuth/elevation update in real time — useful for choosing good observer parameters.

### Create an Animation

```bash
python make_animation.py --seed_setup 42 --seed_g 7 --duration 6
```

---

## Suggested Workflow for Practitioners

### 1. Choose a Gravity Law

Pick a `seed_g` value. This determines the hidden physics you want your network to discover.

```python
from physics_sim import _generate_gravity
grav = _generate_gravity(seed_g=42)
print(f"g(z) = {grav.g0:.4f} + {grav.alpha:.4f} / (z + 10)")
# g(0) ≈ 9.8 m/s², varies with height
```

### 2. Generate Training Data

Create many scenes with different `seed_setup` values but the **same** `seed_g`:

```bash
python generate_training_data.py \
    --seed_g 42 --n_scenes 200 --seed_start 0 \
    --dt 0.2 --t_max 6
```

### 3. Train Your Network

Use the image sequences as input. Each `scene_XXXX/` folder contains a time series of frames rendered from the same viewpoint. Your network should learn to predict `t_{n+1}.png` given `t_0.png ... t_n.png`.

The `params.json` in each scene folder provides ground truth for validation:
- Ball trajectories at each time step
- Initial positions and velocities
- Ramp geometry

### 4. Evaluate

Generate a held-out test set with different `seed_setup` values:

```bash
python generate_training_data.py \
    --seed_g 42 --n_scenes 50 --seed_start 1000 \
    --dt 0.2 --t_max 6 --output_dir test_data
```

Compare your network's predicted frames against the ground truth images. Evaluate whether the network has implicitly learned the gravity law by checking if predicted ball trajectories match the true g(z).

### 5. Generalize

Once your network works for one `seed_g`, test it on a different gravity law:

```bash
python generate_training_data.py \
    --seed_g 99 --n_scenes 50 --output_dir test_new_gravity
```

Can your network detect that the physics has changed? Can it adapt?

---

## API Reference

### `render_scene(seed_setup, seed_g, times, angles, observer_pos, image_size)`

Main rendering function.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed_setup` | int | — | Scene layout seed |
| `seed_g` | int | — | Gravity law seed |
| `times` | float or list | — | Simulation time(s) in seconds |
| `angles` | (az, el) or list | — | Viewing direction(s) in degrees |
| `observer_pos` | (x,y,z) or list | (0,0,2.0) | Camera position(s) |
| `image_size` | (w, h) | (512, 512) | Output resolution |

Returns a list of `H×W×3` uint8 numpy arrays (RGB images, no text overlays).

Broadcasting: if one of `times`, `angles`, `observer_pos` is a single value and the others are lists, the single value is broadcast to match.

### `describe_scene(seed_setup, seed_g)`

Print a human-readable summary of the scene and gravity configuration.

### `get_ball_positions(seed_setup, seed_g, t)`

Return ball positions at time `t` as a list of `(x,y,z)` arrays (or `None` if the ball has left the visible region).

---

## Notes

- **Deterministic**: the same seeds always produce the same scene and images.
- **No network access required**: everything runs locally, offline.
- **Image convention**: output is 1:1 aspect ratio, 512×512 by default. Adjust with `image_size`.
- **Coordinate system**: X-axis left/right, Y-axis forward/back, Z-axis up. Origin is on the ground at the observer's feet.
- **Ball disappearance**: a ball vanishes from the scene if it crosses `x=0` to the other side, simulating it leaving the field of view.
