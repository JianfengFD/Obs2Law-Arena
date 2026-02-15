

### I. Network Architecture (`O2TNet`)

The system is an end-to-end encoder-decoder architecture designed to infer latent physical parameters from high-resolution stereo image sequences and predict future states. It operates on **native resolution inputs** (e.g., ) without artificial resizing to preserve fine details.

#### 1. Input Construction (23 Channels)

The network constructs a dense input tensor representing the system state at two time steps ( and ):

* **Visual Data (12 ch):** Four RGB images: Stereo Left/Right at , and Stereo Left/Right at .
* **Parameter Embeddings (5 ch):** Scalar parameters (time, position, rotation) for the 4 input views and 1 target view are encoded via a `ParamEncoder` into spatial feature maps.
* **Difference Maps (6 ch):** Pairwise grayscale differences between the four visual inputs to explicitly highlight motion and disparity.

#### 2. The Encoder

The encoder maps the 23-channel input to a compact latent vector (, dim=48).

* **Hierarchy:** Uses a 5-level convolutional downsampling structure ().
* **Components:** Each level consists of `ResBlock` units (GroupNorm, SiLU, Conv3x3).
* **Attention:** Multi-head `SelfAttention2d` is applied only at the two deepest levels (lowest resolution) to capture global context.
* **Output:** The final feature map is pooled (`AdaptiveAvgPool`) and projected to a **48-dimensional latent vector**.

#### 3. The Physics Engine (Latent Evolution)

The core module evolves the latent vector  from input state to target future state.

* **Alignment:** The latent vector is rotated into a canonical coordinate system using a learnable orthogonal matrix (implemented via **Cayley transform**).
* **Evolution Loop:** The vector undergoes  steps of evolution. Each step blends two branches:
* **Neural Branch:** A residual MLP processing abstract dynamics.
* **Physics Branch:** Splits the vector into Velocity and Position components. Applies explicit **Newtonian updates** (, ) using learned gravity () and interaction coefficients ().


* **Inverse Alignment:** The evolved vector is rotated back to the original manifold.

#### 4. The Decoder (Updated)

The decoder is a standalone "Hourglass" (U-Net style) network that reconstructs the predicted image. It takes **fresh image inputs** (Left , Left , and their difference) rather than encoder features.

* **Structure:**
* **Down Path:** 4 levels of downsampling.
* **Bottleneck:** ResBlock  Attention  ResBlock at  resolution.
* **Up Path:** 4 levels of upsampling with skip connections from the Down Path.


* **Physics Injection Points:** The evolved physics latent vector () combined with target parameters is injected into the network at **three specific "Middle" locations** via broadcasting:
1. **Middle-Left (Down Path):** Immediately after the first downsampling block (Level 0). Injection happens at **** resolution.
2. **Middle-Center (Bottleneck):** Inside the bottleneck, at the lowest resolution (****).
3. **Middle-Right (Up Path):** Immediately after the third upsampling block (Level 2). Injection happens at **** resolution.



---

### II. Training Infrastructure (`train_o2t.py`)

The training employs an asynchronous Producer-Consumer architecture to decouple rendering from training.

#### 1. Data Generation (Producers)

* **Background Workers:** Multiple CPU threads continuously render synthetic scenes.
* **Sample Validation:** A `has_motion` check discards static samples in dynamic modes to ensure valid training signals.
* **Modes:**
* `DIS_MOD`: Static scenes (depth/disparity).
* `VIEW_MOD`: Camera movement only (3D structure).
* `FUTURE_MOD`: Time evolution (physics dynamics).



#### 2. Replay Buffer

* A thread-safe circular buffer stores up to **5,000 samples**.
* New samples from workers replace the oldest ones.
* The GPU trains on random mini-batches sampled from this buffer.

#### 3. Curriculum Learning

A `TrainingSchedule` manages task complexity based on global steps:

* **0–1k:** Static/Displacement only.
* **1k–10k:** Mixed Static and Viewpoint changes.
* **10k–50k:** Introduces Physics with short horizons ().
* **50k+:** Primarily Physics with increasing horizons ( up to 50).

#### 4. Optimization

* **Loss:** Composite loss of .
* **Physics Weight Schedule:** The blending weight  (Neural vs. Analytical) decays exponentially as the prediction horizon  increases, forcing the model to rely on analytical physics for long-term predictions.