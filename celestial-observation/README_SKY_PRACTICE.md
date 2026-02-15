# Celestial Observation Practice — README

## Project Goal

This project provides a **virtual astronomical observatory**: a simulated solar system with modified gravity, where you stand on the surface of the Earth and observe the sky through a virtual telescope. The system renders realistic sky images showing the Sun, Moon, inner planets, and background stars.

The purpose is to serve as a **practice dataset generator** for researchers building neural networks or analysis pipelines that learn orbital dynamics and physical laws from sky observations. Specifically:

> **Task**: given a sequence of telescope images taken over time, discover the underlying gravity law that governs planetary motion.

In practice mode, the gravity modification parameter `alpha` and Earth's axial tilt `tilt_deg` are known to you, so you have full ground truth for training and validation.

---

## The Physics

### Modified Gravity

The gravitational force between two bodies is:

```
F ∝ (1 + alpha) / r^(2 + alpha)
```

where `alpha` is a small perturbation to Newtonian gravity. When `alpha = 0`, this reduces to standard 1/r² gravity. The parameter `alpha` is set in the simulation program and controls how much the orbits deviate from Keplerian ellipses.

### Solar System Simulation

The simulation (`solar_run_rebound_ME_alpha_data.py`) uses the REBOUND N-body integrator with:

- **10 bodies**: Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Moon
- **Real initial conditions** from JPL ephemeris (DE421), starting from 2020-01-01
- **WHFast symplectic integrator** with 2-hour base time step
- **Output**: positions and velocities of tracked bodies (Sun, Mercury, Venus, Earth, Mars, Moon) sampled every 0.5 hours

### Virtual Telescope

The observer (`earth_view_mod.py`) stands on the Earth's surface and looks through a virtual telescope:

- **Observer position**: specified by longitude and latitude on Earth
- **Telescope pointing**: azimuth (phi) and elevation (theta) angles
- **Field of view**: ~10° half-angle (adjustable via zoom)
- **Background**: Tycho-2 star catalog (~2.5 million stars)
- **Earth axial tilt**: default 23.44° (the obliquity of the ecliptic)
- **Rendered objects**: Sun (yellow), Moon (grey), Mercury (orange), Venus (gold), Mars (red), plus background stars
- **Output**: 1024×1024 circular sky images (PIL Image format)

---

## File Overview

| File | Description |
|---|---|
| `solar_run_rebound_ME_alpha_data.py` | N-body simulation: generates trajectory data |
| `earth_view_mod.py` | VirtualTelescope class: renders sky images from trajectory data |
| `generate_sky_training_data.py` | Practice: batch-generate training images |
| `gen_sealed_telescope.py` | **Organiser tool**: encrypt data + seal telescope for competition |
| `sealed_telescope_XXX.py` | Competition: sealed telescope module (generated) |
| `generate_sky_training_data_sealed.py` | Competition: batch-generate from sealed module |

---

## Quick Start (Practice Mode)

### 1. Install Dependencies

```bash
pip install numpy scipy matplotlib Pillow astropy rebound skyfield
# Download DE421 ephemeris (first run of skyfield handles this)
# Download Tycho-2 catalog: tycho2_entire_sky.fits (from VizieR or your data source)
```

### 2. Generate Trajectory Data

```bash
# Simulate 20 years with default alpha=0.01
python solar_run_rebound_ME_alpha_data.py 20

# Output: simulation_data_20.0yrs.txt
#         solar_system_orbits_20.0yrs.png (orbit plot)
```

To change `alpha`, edit line 89 in the simulation file:
```python
alpha = 0.01  # ← change this value
```

### 3. Render Sky Images

#### Single image (Python)

```python
from earth_view_mod import VirtualTelescope

telescope = VirtualTelescope(
    simulation_data_path="simulation_data_20.0yrs.txt",
    star_catalog_path="tycho2_entire_sky.fits",
)

img = telescope.capture(
    time_days=100.0,     # 100 days after simulation start
    lon_deg=86.0,        # longitude
    lat_deg=0.0,         # latitude (equator)
    phi_deg=-90.0,       # look south
    theta_deg=5.0,       # 5° above horizon
    zoom=1.0,
    tilt_deg=23.44,      # Earth axial tilt
)
img.save("sky_day100.png")
```

#### Batch generation

```bash
python generate_sky_training_data.py \
    --sim_data simulation_data_20.0yrs.txt \
    --n_times 100 --t_max 730 \
    --lon 86 --lat 0 --phi -90 --theta 5
```

Output:
```
sky_training_data/
  metadata.json
  sky_t00000.00.png
  sky_t00007.37.png
  ...
```

#### Multi-angle sweep

```python
# Same time, different viewing directions
imgs = telescope.capture(
    time_days=100.0,
    lon_deg=86.0, lat_deg=0.0,
    phi_deg=[-180, -90, 0, 90, 180],   # full horizon sweep
    theta_deg=5.0, zoom=1.0,
)
for i, img in enumerate(imgs):
    img.save(f"panorama_{i}.png")
```

---

## `capture()` Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `time_days` | float or list | Simulation time (days from start) |
| `lon_deg` | float or list | Observer longitude on Earth (degrees) |
| `lat_deg` | float or list | Observer latitude (degrees, 0=equator) |
| `phi_deg` | float or list | Telescope azimuth (degrees, -90=south) |
| `theta_deg` | float or list | Telescope elevation (degrees, 0=horizon, 90=zenith) |
| `zoom` | float or list | Zoom factor (1.0 = default ~10° half-FOV) |
| `tilt_deg` | float | Earth axial tilt (degrees, default 23.44) |

All parameters except `tilt_deg` support broadcasting: pass lists of different lengths and the shorter ones repeat.

---

## Competition Mode

For organising a competition where `alpha` and `tilt_deg` are hidden:

### Step 1: Generate trajectory data (organiser only)

```bash
python solar_run_rebound_ME_alpha_data.py 20
# Edit alpha as desired before running
```

### Step 2: Seal the telescope

```bash
python gen_sealed_telescope.py \
    --sim_data simulation_data_20.0yrs.txt \
    --alpha 0.01 \
    --tilt 23.439281 \
    --func_id T01
```

This produces:
- `trajectory_T01.enc` — encrypted trajectory data (participants get this)
- `sealed_telescope_T01.py` — sealed module (participants get this)
- `key_telescope_T01.txt` — answer key (organiser keeps this)

### Step 3: Participants generate training data

```bash
python generate_sky_training_data_sealed.py \
    --module sealed_telescope_T01 \
    --n_times 100 --t_max 730
```

Participants can only call `capture_T01(...)` to get images. They cannot access the trajectory data, tilt angle, or alpha value.

---

## What Participants Must Discover

1. **Earth's axial tilt** (`tilt_deg`): affects how the ecliptic plane appears from the observer's location. The seasonal variation of planet positions in the sky encodes the tilt.

2. **Gravity modification** (`alpha`): affects orbital periods, eccentricities, and precession rates. With `alpha ≠ 0`, Mercury's orbit precesses differently than Newtonian prediction. Long time baselines are needed to detect this.

3. **Orbital dynamics**: even without knowing the exact gravity law, predicting where planets will appear in the sky at future times demonstrates understanding of the underlying physics.
