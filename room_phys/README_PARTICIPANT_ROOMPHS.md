# Physics from Vision Challenge — Participant Guide

## What Is This Challenge

You are given a **black-box physics renderer**. It simulates balls moving in a 3D world under an **unknown, height-dependent gravity law** g(z), then renders images from a virtual camera.

**Your task**: by observing how objects move in rendered image sequences, **discover the hidden gravity function g(z)**.

You have no access to ball positions or trajectories in numeric form — only rendered images. You must extract physics from pixels.

---

## What You Receive

| File | Purpose |
|---|---|
| `render_scene_XXX.py` | Sealed renderer (XXX is your problem ID) |
| `generate_training_data_sealed.py` | Batch image generation script |
| `interactive_viewer_sealed.py` | Interactive 3D exploration (optional) |
| This document | Usage guide |

---

## Setup

```bash
pip install numpy pyvista imageio Pillow
# For interactive viewer (optional):
pip install pyvistaqt PyQt5
```

---

## What You Know

The following facts about the physics world are public:

**Gravity** is height-dependent: g depends on z (the altitude above the ground). The functional form and parameters are hidden — that is what you need to discover.

**Scene layout** (each scene has a random seed `seed_setup`):

- 2 balls per scene, one on each side of x = 0.
- Ball radii follow a Gaussian distribution (mean ≈ 0.4 m, clipped to 0.1–1.5 m).
- Each ball is one of: projectile (launched with initial velocity), free-fall (dropped from rest), or ramp (starts from rest on an inclined plane).
- Balls on ramps are rolling solid spheres (effective acceleration factor = 5/7 × g sin θ).
- A ball disappears when it crosses x = 0 to the other side.
- 0 or 1 ramp per scene, always on the X > 0 side.

**Camera**: pin-hole model, fixed 60° field of view, always level (no tilt). Default position (0, 0, 2).

**Visual cues in each image** (no text overlays):

- Ball shadows: dark discs projected vertically onto the ground — tells you approximate (x, y).
- Red scale bars: fixed at (3, 0) and (−3, 0), length = ball diameter. These do not move.
- Ground grid: 1 m spacing.
- Ramp with height tick marks on the side.

---

## Generating Data

### Batch generation (recommended for training)

```bash
# Generate 200 scenes, 11 frames each (0s to 5s, every 0.5s)
python generate_training_data_sealed.py \
    --module render_scene_XXX \
    --n_scenes 200 \
    --dt 0.5 --t_max 5

# Finer time resolution for more detailed motion
python generate_training_data_sealed.py \
    --module render_scene_XXX \
    --n_scenes 100 \
    --dt 0.1 --t_max 4 \
    --seed_start 1000

# Custom observer position and viewing angle
python generate_training_data_sealed.py \
    --module render_scene_XXX \
    --n_scenes 50 \
    --obs 5 0 3 --az 180 --el -10
```

Output:

```
training_data/
  metadata.json             ← observer position, angles, time list
  scene_0000/
    t_0.00.png              ← 512×512 RGB image at t=0
    t_0.50.png
    ...
  scene_0001/
    ...
```

### Python API

```python
from render_scene_XXX import render_scene_XXX, describe_scene_XXX

# Render image sequence for one scene
images = render_scene_XXX(
    seed_setup=42,                   # scene layout seed
    times=[0, 0.5, 1.0, 2.0, 3.0],  # time in seconds
    angles=(25, -5),                 # (azimuth, elevation) degrees
)
# images: list of 512×512×3 uint8 numpy arrays

# Inspect scene layout (does NOT reveal gravity)
describe_scene_XXX(seed_setup=42)
# Output: ball types, radii, initial positions/velocities, ramp geometry
```

#### `render_scene_XXX` parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `seed_setup` | int | — | Scene layout seed |
| `times` | float or list | — | Simulation time(s) in seconds |
| `angles` | (az, el) or list | — | Viewing direction in degrees |
| `observer_pos` | (x,y,z) or list | (0,0,2) | Camera position |
| `image_size` | (w, h) | (512,512) | Output image resolution |

Azimuth: 0 = looking along +X, 90 = along +Y. Elevation: 0 = horizontal, positive = up. Broadcasting: a single value is repeated to match lists in other parameters.

#### `describe_scene_XXX` output example

```
Scene seed_setup=42
  Balls: 2  Ramps: 1
  Ball 0 [X>0] projectile r=0.494m
    pos0=[17.335 -0.927  1.787]  vel0=[ 4.41  -1.387  6.722]
  Ball 1 [X<0] freefall r=0.314m
    pos0=[-7.415 12.85   2.086]  vel0=[0. 0. 0.]
  Ramp 0: slope=49.0 deg orient=23.1 deg
    h=1.81m w=2.79m base=(9.30,17.07)
```

This gives you the scene geometry. It does **not** give you ball positions at later times or any gravity information.

### Interactive viewer (optional)

```bash
python interactive_viewer_sealed.py --module render_scene_XXX --seed_setup 42
```

Rotate with mouse, play/pause animation, scrub through time. Useful for building intuition — you see exactly what the renderer produces.

---

## Strategy Hints

### The core difficulty

You only observe images. You must figure out:

1. **Where are the balls?** — Estimate 3D ball positions from 2D images, accounting for camera perspective.
2. **How are they moving?** — Track positions across time to reconstruct trajectories.
3. **What is g(z)?** — From trajectories, infer the gravity law.

Steps 1–2 are computer vision problems. Step 3 is physics inference.

### Useful approaches

**Multi-view rendering**: you can render the same scene from different observer positions and angles. This gives you stereo/multi-view information to triangulate 3D positions.

```python
# Same scene, same time, three viewpoints
images = render_scene_XXX(
    seed_setup=42,
    times=1.0,
    angles=[(0, -5), (90, -5), (45, -30)],
    observer_pos=[(0,0,2), (0,0,2), (0,0,5)],
)
```

**Shadow analysis**: the ground shadow directly tells you the ball's (x, y) position. Combined with the ball's apparent size in the image and its known radius (from `describe_scene_XXX`), you can estimate z.

**Free-fall scenes are simplest**: a ball dropped from rest falls purely vertically. Its z(t) trajectory directly reflects g(z). Filter scenes by `describe_scene_XXX` to find free-fall balls.

**Scale bars as calibration**: the red bars have known length (= ball diameter) and known positions ((±3, 0, 0.1)). Use them to calibrate your image-to-world coordinate mapping.

**Dense time sampling**: render at very fine time steps (dt = 0.02) to get smooth trajectories for numerical differentiation.

```python
import numpy as np
images = render_scene_XXX(
    seed_setup=42,
    times=np.arange(0, 3, 0.02).tolist(),
    angles=(25, -5),
)
```

### What won't work

- There is no `get_ball_positions` function. You cannot bypass the vision problem.
- The gravity law is baked into compiled bytecode. Reverse-engineering the module is against the rules.

---

## Submission Format

Create a file named `submit_room.txt` with the following format:

```
# Room Physics Challenge Submission
[a1b]
g0 = 9.78
alpha = -1.2
```

where `a1b` is the problem ID (from `render_scene_a1b.py`), and `g0`, `alpha` are your estimates for the gravity law:

```
g(z) = g0 + alpha / (z + 10)
```

If the competition has multiple problems, include one block per problem:

```
[a1b]
g0 = 9.78
alpha = -1.2

[c3f]
g0 = 9.5
alpha = 3.0
```

### Scoring

Your score is computed by `score_room.py` (provided by the organiser):

```bash
python score_room.py --submission submit_room.txt --keys key_a1b.txt key_c3f.txt
```

The score combines two metrics (lower is better):

1. **Functional match (weight 0.7)**: mean |g_yours(z) − g_true(z)| over z ∈ [0, 20], step 0.1. This measures how well your predicted gravity curve matches the truth.

2. **Parameter match (weight 0.3)**: |g0_err|/|g0_true| + |alpha_err|/max(|alpha_true|, 0.1). This measures how close your raw parameters are.

```
total_score = 0.7 × functional_error + 0.3 × parameter_error
```

For multi-problem competitions, the final score is the **average** across all problems.

---

## Rules

- You may generate unlimited data using `render_scene_XXX`.
- You may use any tools, libraries, or methods (classical CV, deep learning, etc.).
- You may render from any observer position and angle.
- You may use `describe_scene_XXX` to inspect scene layouts.
- You may **not** reverse-engineer the sealed module to extract parameters.
- If unsure whether something is allowed, ask the organiser.

---

## FAQ

**Q: Can I get exact ball coordinates?**
A: No. You only get rendered images. Extracting 3D information from images is part of the challenge.

**Q: Is gravity constant?**
A: No. It depends on height z. Discovering the form of g(z) is the goal.

**Q: Do all scenes share the same gravity?**
A: Yes. Only scene layout changes with `seed_setup`. The gravity law is fixed.

**Q: How many scenes should I generate?**
A: For exploration, 10–20 scenes are enough. For serious estimation, 100+ scenes with fine time steps.

**Q: Can I render from any viewpoint?**
A: Yes. Choosing smart viewpoints (e.g., side view for tracking z, top-down for x-y) is a key part of the challenge.

**Q: What is the coordinate system?**
A: Right-handed. X = left/right, Y = forward/back, Z = up. Origin on the ground.
