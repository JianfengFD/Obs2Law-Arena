# Celestial Observation Challenge — Participant Guide

## What Is This Challenge

You have access to a **sealed virtual telescope**. It simulates a solar system where gravity has been subtly modified, and lets you observe the sky from any point on Earth's surface at any time over a multi-year period.

**Your task**: by analysing sequences of sky images, discover:

1. **The gravity law modification** — how does gravity differ from the standard 1/r² Newtonian law?
2. **Earth's axial tilt** — what is the obliquity of the ecliptic in this simulated world?

You have no access to planet positions or orbital data — only rendered telescope images.

---

## What You Receive

| File | Purpose |
|---|---|
| `sealed_telescope_XXX.py` | Sealed telescope module (XXX = problem ID) |
| `trajectory_XXX.enc` | Encrypted orbital data (required, do not modify) |
| `generate_sky_training_data_sealed.py` | Batch image generation script |
| `tycho2_entire_sky.fits` | Background star catalog |
| This document | Usage guide |

---

## Setup

```bash
pip install numpy scipy matplotlib Pillow astropy cryptography
```

Place `trajectory_XXX.enc` and `tycho2_entire_sky.fits` in the same directory as the sealed module.

---

## What You Know

**The solar system** contains the Sun, Mercury, Venus, Earth, Mars, and the Moon. Their real initial conditions (from 2020-01-01) were used, then evolved forward under a **modified gravity law** for many years.

**The gravity law** has the form:

```
F ∝ (1 + alpha) / r^(2 + alpha)
```

where `alpha` is a small unknown parameter. When `alpha = 0`, this is standard Newtonian gravity. Your task is to determine `alpha`.

**Earth's axial tilt** (`tilt_deg`) controls how the ecliptic plane is tilted relative to the equator. It affects the apparent seasonal motion of the Sun and planets across the sky. The true value is hidden — you need to measure it.

**The telescope** renders 1024×1024 circular sky images showing:
- Sun (yellow disc), Moon (grey), Mercury (orange), Venus (gold), Mars (red)
- Background stars from the Tycho-2 catalog
- A brown strip at the bottom for the ground/horizon
- The field of view is approximately 10° half-angle (adjustable via zoom)

---

## Generating Data

### Batch generation

```bash
# 100 images over one year
python generate_sky_training_data_sealed.py \
    --module sealed_telescope_XXX \
    --n_times 100 --t_max 365 \
    --lon 86 --lat 0 --phi -90 --theta 5

# 50 images over 5 years, different observer
python generate_sky_training_data_sealed.py \
    --module sealed_telescope_XXX \
    --n_times 50 --t_start 0 --t_max 1826 \
    --lon 0 --lat 45 --phi 0 --theta 30
```

Output:
```
sky_training_data/
  metadata.json
  sky_t00000.00.png
  sky_t00003.69.png
  ...
```

### Python API

```python
from sealed_telescope_XXX import capture_XXX, info_XXX

# Check available time range
info_XXX()

# Single image
img = capture_XXX(
    time_days=100.0,      # simulation time (days from start)
    lon_deg=86.0,         # observer longitude
    lat_deg=0.0,          # observer latitude (0 = equator)
    phi_deg=-90.0,        # telescope azimuth (-90 = south)
    theta_deg=5.0,        # telescope elevation above horizon
    zoom=1.0,             # zoom factor
)
img.save("sky_day100.png")  # PIL Image

# Multiple images at different times
imgs = capture_XXX(
    time_days=[0, 30, 60, 90, 120],
    lon_deg=86.0, lat_deg=0.0,
    phi_deg=-90.0, theta_deg=5.0,
)
for i, im in enumerate(imgs):
    im.save(f"frame_{i}.png")

# Multi-angle panorama at one time
imgs = capture_XXX(
    time_days=100.0,
    lon_deg=86.0, lat_deg=0.0,
    phi_deg=[-180, -135, -90, -45, 0, 45, 90, 135, 180],
    theta_deg=5.0,
)
```

### Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `time_days` | float or list | Simulation time (days from epoch) |
| `lon_deg` | float or list | Observer longitude on Earth (degrees) |
| `lat_deg` | float or list | Observer latitude (degrees, 0=equator) |
| `phi_deg` | float or list | Telescope azimuth (degrees) |
| `theta_deg` | float or list | Telescope elevation (degrees, 0=horizon) |
| `zoom` | float or list | Zoom factor (1.0 = ~10° half-FOV) |

Lists of different lengths broadcast (shorter ones repeat).

**Note**: there is no `tilt_deg` parameter. The tilt is fixed and hidden inside the module.

---

## Strategy Hints

### Measuring the axial tilt

The Sun's declination (its angular height above/below the celestial equator) oscillates over a year between +tilt and −tilt. By tracking the Sun's maximum elevation at solar noon from the equator across a full year, you can directly measure the axial tilt.

**Approach**: render images at the equator (`lat=0`) looking south (`phi=-90`) at noon-equivalent times, and track the Sun's vertical position in the image across months.

### Measuring alpha

`alpha` affects orbital dynamics in subtle ways:

- **Orbital precession**: Mercury's perihelion precesses. With `alpha ≠ 0`, the precession rate differs from the Newtonian prediction. This requires long time baselines (years) and precise position tracking.
- **Orbital periods**: slightly different from Keplerian values.
- **Eccentricity evolution**: non-Newtonian gravity causes orbits to evolve over time.

**Approach**: track a planet's position over many orbits. Fit an ellipse to each orbit and measure how the ellipse rotates between orbits. The precession rate encodes `alpha`.

### General tips

- **Multi-view triangulation**: render the same time from different longitudes to triangulate planet positions in 3D.
- **Dense time sampling**: use dt = 0.5 days or finer to track fast-moving bodies (Moon, Mercury).
- **Background stars as reference**: the star field provides a fixed reference frame. Planet motion relative to stars gives angular velocities.
- **Long baselines for alpha**: you may need 5–20 years of simulated data to detect small alpha values.
- **Zoom in**: use `zoom=2` or higher to resolve small angular separations.

### What won't work

- There is no function to get planet coordinates directly.
- The trajectory data file is AES-encrypted.
- The axial tilt is not accessible as a parameter.
- Reverse-engineering the sealed module is against the rules.

---

## Submission Format

Create a file named `submit_sky.txt` with the following format:

```
# Celestial Observation Challenge Submission
[T01]
alpha = 0.012
tilt = 23.5
```

where `T01` is the problem ID (from `sealed_telescope_T01.py`), and `alpha`, `tilt` are your estimates.

If the competition has multiple problems, include one block per problem:

```
[T01]
alpha = 0.012
tilt = 23.5

[T02]
alpha = 0.005
tilt = 24.1
```

### Scoring

Your score is computed by `score_sky.py` (provided by the organiser):

```bash
python score_sky.py --submission submit_sky.txt --keys key_telescope_T01.txt key_telescope_T02.txt
```

The score combines two metrics (lower is better):

1. **Alpha accuracy (weight 0.8)**: |alpha_sub − alpha_true| / |alpha_true|. Relative error of the gravity modification parameter.

2. **Tilt accuracy (weight 0.2)**: |tilt_sub − tilt_true| in degrees. Absolute error of the axial tilt.

```
total_score = 0.8 × alpha_relative_error + 0.2 × tilt_absolute_error_deg
```

For multi-problem competitions, the final score is the **average** across all problems.

---

## Rules

- You may generate unlimited sky images.
- You may observe from any location, direction, time, and zoom level.
- You may use any analysis tools (CV, ML, classical astronomy, etc.).
- You may **not** reverse-engineer the sealed module or decrypt the trajectory file.
- If unsure whether something is allowed, ask the organiser.

---

## FAQ

**Q: Can I get planet coordinates numerically?**
A: No. You only get rendered images. Extracting positions from pixels is part of the challenge.

**Q: What is the time range?**
A: Call `info_XXX()` to see. Typically 0 to several thousand days (years of data).

**Q: Do background stars move?**
A: No. Stars are fixed (at their Tycho-2 catalog positions). Only solar system bodies move.

**Q: What coordinate system is used?**
A: ICRS (equatorial). The axial tilt rotates this into the ecliptic frame. You observe in a local horizon frame defined by your longitude, latitude, and telescope pointing.

**Q: Is alpha the same for all body pairs?**
A: Yes. The same gravity law applies universally.
