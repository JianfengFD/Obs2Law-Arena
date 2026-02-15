# Celestial Observation Challenge — Organiser Guide

## Overview

`gen_sealed_telescope.py` generates a **sealed virtual telescope** for astronomical observation competitions. The gravity modification parameter `alpha` and Earth's axial tilt `tilt_deg` are hidden inside the module; participants must discover them by analysing sky images taken over time.

---

## Quick Start

### 1. Generate trajectory data

```bash
# Edit alpha on line 89 of the simulation script before running
# alpha = 0.01 (default)
python solar_run_rebound_ME_alpha_data.py 20
# → simulation_data_20.0yrs.txt
```

### 2. Seal the telescope

```bash
python gen_sealed_telescope.py \
    --sim_data simulation_data_20.0yrs.txt \
    --alpha 0.01 \
    --tilt 23.439281 \
    --func_id T01
```

Output:

| File | Who gets it |
|---|---|
| `trajectory_T01.enc` | Participants (encrypted, unreadable) |
| `sealed_telescope_T01.py` | Participants (black-box renderer) |
| `key_telescope_T01.txt` | **Organiser only** (answer key) |

### 3. Distribute to participants

Give them:
- `trajectory_T01.enc`
- `sealed_telescope_T01.py`
- `generate_sky_training_data_sealed.py`
- `tycho2_entire_sky.fits` (background star catalog)
- The participant README

---

## What Participants Get

A single function:

```python
from sealed_telescope_T01 import capture_T01

img = capture_T01(
    time_days=100.0,
    lon_deg=86.0, lat_deg=0.0,
    phi_deg=-90.0, theta_deg=5.0,
    zoom=1.0,
)
img.save("sky.png")  # PIL Image, 1024×1024
```

They **cannot**:
- Read or decrypt trajectory data
- Access planet positions numerically
- Read `alpha` or `tilt_deg` from the source file
- Pass a custom `tilt_deg` to the capture function

They **can**:
- Render unlimited sky images at any time, location, and pointing direction
- Call `info_T01()` to see tracked body names and time range
- Use any analysis method on the resulting images

---

## Security Analysis

### Three layers of protection

| Layer | Technique | What it blocks |
|---|---|---|
| **Data encryption** | Fernet (AES-128-CBC) on trajectory file | Reading planet positions from the `.enc` file |
| **Bytecode sealing** | `earth_view_mod.py` + secrets compiled → marshal → zlib → base64 | `grep` / source inspection for alpha, tilt, passphrase |
| **Closure isolation** | `alpha`, `tilt`, decryption key are local variables in `_build()`, deleted from namespace after construction | `_W["_TILT"]`, `module._ALPHA` etc. all blocked |

### Known attack vectors

| Attack | Difficulty | Mitigation |
|---|---|---|
| **Closure inspection** on `_capture_fn.__closure__` | Medium | Tilt is float inside VirtualTelescope instance, not a simple closure cell. Attacker would need to navigate multiple object layers. |
| **Bytecode disassembly** of the blob | Medium-Hard | Standard marshal + zlib. Consider distributing `.pyc` only for higher security. |
| **Brute-force tilt**: render with known tilt vs sealed output and compare | Hard | Tilt is a continuous float (not an integer seed). The search space is infinite. |
| **Brute-force alpha**: simulate orbits with different alphas and match planet positions | Hard | Requires running REBOUND + matching against extracted pixel positions. Very labour-intensive — which is essentially solving the challenge. |

### Extra hardening options

**Distribute .pyc only:**
```bash
python -c "import py_compile; py_compile.compile('sealed_telescope_T01.py')"
# Give participants the .pyc, delete the .py
```

**Perturb tilt slightly:**
```bash
# Use a non-standard tilt to make brute-force harder
python gen_sealed_telescope.py --tilt 23.5 --alpha 0.015 --func_id T02
```

---

## Generating Multiple Problems

```bash
# Problem A: standard gravity, standard tilt
python gen_sealed_telescope.py --alpha 0.0 --tilt 23.439281 --func_id A01 \
    --sim_data simulation_alpha0.0_20yrs.txt

# Problem B: modified gravity, standard tilt
python gen_sealed_telescope.py --alpha 0.01 --tilt 23.439281 --func_id B01 \
    --sim_data simulation_alpha0.01_20yrs.txt

# Problem C: modified gravity + modified tilt
python gen_sealed_telescope.py --alpha 0.02 --tilt 24.0 --func_id C01 \
    --sim_data simulation_alpha0.02_20yrs.txt
```

Each requires a separate REBOUND simulation run with the corresponding `alpha`.

---

## Answer Verification

Use `score_sky.py` to evaluate participant submissions:

```bash
# Single problem
python score_sky.py --submission submit_sky.txt --keys key_telescope_T01.txt

# Multiple problems
python score_sky.py --submission submit_sky.txt --keys key_telescope_T01.txt key_telescope_T02.txt

# Or point to a directory of key files
python score_sky.py --submission submit_sky.txt --key_dir ./keys/
```

Submission file format (`submit_sky.txt`):
```
[T01]
alpha = 0.012
tilt = 23.5

[T02]
alpha = 0.005
tilt = 24.1
```

Score formula:
```
score = 0.8 × |alpha_sub - alpha_true| / |alpha_true|
      + 0.2 × |tilt_sub - tilt_true|  (degrees)
```

Weights can be adjusted via `--w_alpha` and `--w_tilt`.

---

## Dependencies (for organiser)

```bash
pip install numpy scipy matplotlib Pillow astropy rebound skyfield cryptography
```

The `cryptography` package provides Fernet (AES) encryption. If unavailable, `gen_sealed_telescope.py` falls back to XOR obfuscation (less secure but zero extra dependencies).
