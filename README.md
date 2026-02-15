# Obs2Law-ARENA

**Not related to Science-Discovery/AI-Newton.** (This repo is independent and not a subproject, fork, or derivative.)

Obs2Law-ARENA is a shared, physics-grounded training arena for AI4S: an agent queries **where it is** and **how it looks**, and receives **rendered observations** (images) instead of privileged state.
We bridge two regimes—**room-scale experiments** and **millennia-scale planetary dynamics**—under a unified “observation → law” goal.
Historically, discovering gravity was hard precisely because humans didn’t start from clean trajectory tables: we first had to build instruments, fight noise, and remove confounders.
Even after getting numbers, we still had to factor out viewpoint, reference frames, Earth rotation, and seasonal geometry before any law-fitting began.
Many benchmarks begin after these steps; Obs2Law-ARENA starts closer to the observational reality—**from pixels to laws**.

---

## What this repository is for

Obs2Law-ARENA is designed to be a **public, reproducible benchmark + training ground** for:
- **Observation-to-Theory (O2T)** learning: infer hidden physical laws from visual observations.
- **AI4S competitions**: provide a common environment where results are comparable.
- **Baseline model development**: include/host reference pipelines that operate on the same observation interface.

Core principle: **you don’t get the state**. You get images (and minimal public metadata), and you control *when/where/how* you observe.

---

## Two benchmark worlds

### 1) Room Physics (RoomPhys): indoor gravity from vision
A controlled 3D world where balls undergo projectile motion, free-fall, and ramp motion under a **hidden, height-dependent gravity law**:

- **Goal (competition mode)**: infer the unknown gravity function from rendered image sequences.
- **Observation**: 512×512 RGB frames from a pin-hole camera; you choose camera position and viewing angles.
- **Scene**: two balls, optional ramp, ground grid, shadows, and scale bars for calibration.

**Practice mode** provides full ground truth (for training/validation).
**Competition mode** provides a sealed black-box renderer (for benchmark fairness).

Typical tasks:
- recover gravity parameters / curve g(z)
- predict future frames under the same hidden physics
- compare methods under identical observation controls

---

### 2) Celestial Observation (Sky): planetary dynamics from a virtual telescope
A sealed “virtual observatory” where you stand on Earth and observe a simulated solar system over years.

- **Goal (competition mode)**: discover hidden physical parameters of the world from sky images only:
  - a modified gravity law parameter (deviation from Newtonian 1/r²)
  - Earth’s axial tilt (obliquity) in that simulated world
- **Observation**: 1024×1024 circular sky images (telescope view) including Sun/Moon/inner planets + background stars.
- **Control**: you choose time, longitude/latitude on Earth, telescope azimuth/elevation, and zoom.

As with RoomPhys:
- **Practice mode** exposes ground truth for learning and debugging.
- **Competition mode** seals the true parameters and orbital data.

---

## Baseline attempt included: O2T Network (Observation → Theory)
This repo also documents (and may include) a reference “Observation-to-Theory” neural architecture designed to:
- operate on native high-resolution inputs (e.g., 512×512) without pre-resize,
- use an explicit physics-informed bottleneck (learnable parameters forced toward physical meaning),
- train with a replay-buffer pipeline (CPU renders, GPU trains) and a curriculum from reconstruction → viewpoint disentanglement → future prediction.

The intent is not to declare a “winner,” but to provide a **starting baseline** that others can reproduce and improve.

---

## Why Obs2Law (and why images)?
In real scientific discovery, the hardest part is often upstream of curve fitting:
- turning messy observations into reliable measurements,
- separating instrument/camera effects from dynamics,
- accounting for changing frames (rotation, tilt, perspective) and environmental confounders.

Obs2Law-ARENA makes those “pre-theory” steps part of the benchmark:
you must earn your trajectory tables (if you want them) from images, under controlled but nontrivial observation geometry.

---

## How to use
Pick a world and follow its README:
- **RoomPhys**: generate scenes and image sequences; train models; evaluate on sealed modules.
- **Sky**: query a virtual telescope across time and locations; infer hidden parameters.

The repository is intended to be modular:
- environment generators / sealed modules,
- data generation scripts,
- scoring / evaluation utilities,
- baseline O2T implementations.

---

## License
This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

---

## Citation / attribution
If you use Obs2Law-ARENA in papers, reports, or competition writeups, please cite the repository and keep attribution in derived datasets and forks.

---

## Disclaimer
This repository is **NOT related to Science-Discovery/AI-Newton**. Names that resemble “AI-Newton” refer only to the general idea of rediscovering physics from observation, not to that specific project.
