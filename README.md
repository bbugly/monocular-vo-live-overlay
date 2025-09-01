# Monocular VO — Live Overlay + Absolute Scale

This project provides **`flexible_tube_cam_tracking.py`**, a lightweight Python script that estimates **camera motion from a single video** (monocular VO) and draws a **live overlay** showing:

- **Tracked feature points** (green; bright = inliers used by pose, red = others)
- **Camera axes at the current pose** — **X**=blue, **Y**=green, **Z**=red
- **Origin marker “O”** — the current camera position `t_WC`
- **HUD (top‑right)** — `good` match count and `(x, y, z)`
- **Frame index (bottom‑right)** — small 4‑digit counter

It also exports the **trajectory** as `.npy` (and optional `.csv`) and can plot a **unit‑aware 3D trajectory**.

---

## How it works (algorithm)

1. **Corners**
   Detect Shi–Tomasi features with `cv2.goodFeaturesToTrack`.

2. **Tracking**
   Track features between frames using pyramidal Lucas–Kanade (`cv2.calcOpticalFlowPyrLK`).

3. **Relative pose (prev → curr)**
   Estimate the Essential matrix with RANSAC and recover pose:

   * `E = cv2.findEssentialMat(prev_points, curr_points, K, method=cv2.RANSAC, ...)`
   * `R, t = cv2.recoverPose(E, prev_points, curr_points, K)`
   * Important: the order is **(prev_points, curr_points)**.

4. **Pose integration (internal state = Camera-from-World)**
   The VO state is accumulated in the camera/body frame as:

   * `R_CW[k] = R_k @ R_CW[k-1]`
   * `t_CW[k] = R_k @ t_CW[k-1] + t_k`
     (This represents X_cam = R_CW * X_world + t_CW.)

5. **World pose for visualization/outputs**
   Convert to World-from-Camera when drawing or exporting:

   * `R_WC = R_CW.T`
   * `t_WC = - R_CW.T @ t_CW`
     The world origin is the first frame’s camera pose.

6. **Scale (optional, but recommended for meters)**
   Monocular VO has unknown absolute scale. Apply a constant factor:

   * CLI: `--scale S`
   * All HUD/console/CSV/plot outputs use the same scaled units.

7. **Overlay**
   Draw tracked points (green = inliers, red = outliers), current camera axes at the current position, the origin marker “O”, a top-right HUD with `good` and `(x, y, z)`, and a bottom-right frame index.

---

## Installation
- Python 3.8+
- Packages: `numpy`, `opencv-python`, `matplotlib` (only needed for `--plot`)

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install numpy opencv-python matplotlib
```

---

## Camera intrinsics (K, dist)
Provide an `.npz` with:
- `K`: 3×3 camera matrix (float32)
- `dist`: distortion coefficients (e.g., 1×5, float32)

---

## Updating intrinsics

This repository also provides **`update_intrinsics.py`** for quick editing of your calibration file (`intrinsics.npz`).  
It shows how to update `K` (fx, fy, cx, cy) and reset `dist` if needed.

Example usage from `update_intrinsics.py`:
```python
import numpy as np

data = np.load("intrinsics.npz")
K = data['K'].copy()

# Update intrinsics
K[0, 0] = 914.01   # fx: focal length in pixels along x-axis
                   #     fx = (F * px) / sx
                   #     F = focal length in mm
                   #     px = image width in pixels
                   #     sx = sensor width in mm

K[1, 1] = 514.13   # fy: focal length in pixels along y-axis
                   #     fy = (F * py) / sy
                   #     F = focal length in mm
                   #     py = image height in pixels
                   #     sy = sensor height in mm

K[0, 2] = 640.00   # cx: principal point x-coordinate (usually image_width / 2)
K[1, 2] = 360.00   # cy: principal point y-coordinate (usually image_height / 2)

# Define distortion coefficients (here: no distortion)
dist = np.array([0., 0., 0., 0., 0.], dtype=np.float32)

np.savez("intrinsics.npz", K=K, dist=dist)
print("K =\n", K)
print("dist =", dist)
```

### Notes
- `K` is the **intrinsic camera matrix**:
  ```
  [ fx   0  cx ]
  [  0  fy  cy ]
  [  0   0   1 ]
  ```
- `dist` are the **distortion coefficients**: `[k1, k2, p1, p2, k3]` (radial + tangential).  
- If your lens is nearly ideal, just set all `dist` terms to zero.

In OpenCV, the distortion parameter array dist = [k1, k2, p1, p2, k3] is typically interpreted as:
k1, k2, k3 → Radial distortion coefficients
They correct barrel distortion (image bulges outward) and pincushion distortion (image pinches inward).
k1 has the strongest effect because it’s the first-order term; it controls the main curvature of the distortion.
k2 and k3 are higher-order terms (4th-order and 6th-order, respectively). They act as fine-tuning for more subtle distortions after k1.
p1, p2 → Tangential distortion coefficients
They correct distortion caused by the lens not being perfectly perpendicular to the image sensor.
Usually very small (near zero), but they matter if the lens is tilted relative to the sensor.

k1 → main adjustment (largest influence)
k2, k3 → progressively finer adjustments (small influence)
p1, p2 → correct for sensor/lens misalignment

---

## Run it (Terminal & Bash)
Replace paths with yours.

### macOS / Linux (bash/zsh)
```bash
python flexible_tube_cam_tracking.py   --video keyboard.mp4   --intrinsics intrinsics.npz   --frames 500   --csv traj_500.csv   --plot
```

### Windows (PowerShell)
```powershell
python flexible_tube_cam_tracking.py `
  --video keyboard.mp4 `
  --intrinsics intrinsics.npz `
  --frames 500 `
  --csv traj_500.csv `
  --plot
```

### Windows (cmd.exe)
```bat
python flexible_tube_cam_tracking.py --video keyboard.mp4 --intrinsics intrinsics.npz --frames 500 --csv traj_500.csv --plot
```

> Webcam input: pass the device index, e.g. `--video 0`.

---

## CLI options

| Option         | Type    | Default          | Description                                             |
|----------------|--------:|:----------------:|---------------------------------------------------------|
| `--video`      | str/int | **required**     | Input video path or webcam index                        |
| `--intrinsics` | str     | **required**     | `.npz` with `K` and `dist`                              |
| `--frames`     | int     | `0`              | Max frames (`0` = all)                                  |
| `--scale`      | float   | `None`           | Multiply all **saved/plot/export** translations by this |
| `--axis-len`   | float   | `0.2`            | Length of axes drawn on the frame (visual only)         |
| `--no-show`    | flag    | off              | Don’t open the live window                              |
| `--plot`       | flag    | off              | Show a 3D trajectory plot after processing              |
| `--csv`        | str     | `None`           | Save CSV trajectory (`N×3`: x,y,z)                      |
| `--output`     | str     | `trajectory.npy` | Save NumPy trajectory (`3×N`)                           |

**Window controls**: `q` quit, `p` pause/resume.

---

## Live overlay & coordinate system

- **Tracked points**: **green = inliers used by pose**, **red = outliers**.
- **Axes**: the **current camera axes** are drawn at the **current camera position** (they **move with the camera**).  
- **Origin “O”**: white dot at `t_WC`, labeled “O”.  
- **HUD (top-right)**: `good: #### | x:… y:… z:…` shows the **world-frame** position `t_WC` (scaled if `--scale`).  
- **Frame index (bottom-right)**: small 4-digit counter.  
- World origin is **the first frame’s pose**; the drawn axes are **not** fixed world axes (they’re the camera’s local axes).

---

## Scaling to meters (unified units)

Monocular VO gives the **direction** of `t` but not the **scale**. Use `--scale S` to express trajectory in meters. This script also **aligns units across all outputs**:

- **Console & HUD** show `vo.t_WC * S` (meters if `--scale` is given; otherwise arbitrary units).
- **CSV** header becomes `x(m),y(m),z(m)` when scaled.
- **Plot** axis labels include `[m]` (or `[arb. units]` if unscaled).

**How to choose `S`**
- If the real translation across a segment is `D_real` meters and raw VO length is `D_vo`, set `S = D_real / D_vo`.
- Or derive `S` from a known object size / scene geometry.

---

## Outputs
- `trajectory.npy` — `3×N` stacked `[x;y;z]`
- CSV (`--csv`) — `N×3` with unit‑aware header
- 3D plot (`--plot`) — unit‑aware labels

---

## Tips & troubleshooting
- **Window appears zoomed** — Uses `cv2.WINDOW_AUTOSIZE` (1:1 pixels). Avoid `resizeWindow` with AUTOSIZE.  
- **Few tracks (`good` low)** — Improve texture/lighting or tune corner params. `--axis-len` is visualization‑only.  
- **Drift/jitter** — Natural for monocular VO. Use better footage or sensor fusion (IMU/depth). Set absolute scale with `--scale`.  
- **Wrong intrinsics** — Bad `K/dist` destabilizes pose; recalibrate carefully.

---

## FAQ
**Where is the world origin?** First frame camera pose `(0,0,0)`.  
**Are on‑screen axes world‑fixed?** No — they are the **current camera axes**.  
**Webcam?** Yes: `--video 0`.  
**HUD in meters from the start?** Yes — pass `--scale S` and HUD/console/CSV/plot all show meters.
**Which order do I pass points to `findEssentialMat/recoverPose`?**  
Use `(prev_points, curr_points)` (previous → current). The returned `(R, t)` is the relative motion from the previous to the current frame.
**What frame is used to accumulate pose internally?**  
The pose is accumulated as **Camera-from-World** (`T_CW`). For drawing/exports we convert to **World-from-Camera** (`T_WC`), so HUD/CSV/plot all show world-frame `t_WC`.

---
