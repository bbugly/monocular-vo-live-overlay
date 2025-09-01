#!/usr/bin/env python3  # Ensure the script runs with Python 3 when executed directly

# Monocular VO + Absolute Scale + Live Overlay (Fixed for body/world frames)
# -----------------------------------------------------------------------------
# This program performs monocular visual odometry (VO) from a video or webcam
# feed. It tracks features, estimates the Essential matrix with RANSAC, recovers
# relative pose (R, t) between consecutive frames, and composes poses over time.
# The internal pose state is accumulated as Camera-from-World (T_CW); when
# visualizing and exporting we convert to World-from-Camera (T_WC). A live HUD
# overlay shows tracked points, current axes, and position, and the trajectory
# can be saved as .npy/.csv and optionally plotted in 3D.
# -----------------------------------------------------------------------------

import argparse            # Command-line interface argument parsing
import sys                 # System-specific parameters and functions (e.g., sys.exit)
import time                # Timing utilities for FPS/elapsed time measurements
from pathlib import Path   # Object-oriented, safe filesystem path handling
from collections import deque  # Efficient queue for accumulating 3x1 trajectory columns

import numpy as np         # Numerical computing and linear algebra
import cv2                 # OpenCV: computer vision algorithms and I/O


# =====================================
# Function: load_intrinsics
# =====================================

def load_intrinsics(path: Path):
    """
    Load camera intrinsics from a NumPy .npz archive.

    Parameters
    ----------
    path : Path
        Filesystem path to an .npz file that contains:
          - "K": 3x3 camera intrinsic matrix (standard pinhole model layout)
          - "dist": distortion coefficients (shape (N,) or (1,N): e.g., k1,k2,p1,p2,k3)

    Returns
    -------
    K : np.ndarray, dtype float32, shape (3,3)
        Intrinsic calibration matrix (fx, 0, cx; 0, fy, cy; 0, 0, 1).
    dist : np.ndarray, dtype float32, shape (1,N) or (N,)
        Lens distortion coefficients compatible with OpenCV APIs.

    Notes
    -----
    - Raises a KeyError if the required keys are not found.
    - Casting to float32 avoids unnecessary conversions inside OpenCV.
    """
    data = np.load(str(path))            # Open the .npz container (string path for compatibility)
    K = data["K"].astype(np.float32)     # Retrieve intrinsics and ensure float32 dtype
    dist = data["dist"].astype(np.float32)  # Retrieve distortion vector and ensure float32
    return K, dist                       # Return both arrays for downstream use


# =====================================
# Function: rodrigues_from_R
# =====================================

def rodrigues_from_R(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a Rodrigues rotation vector.

    Parameters
    ----------
    R : np.ndarray, shape (3,3)
        Rotation matrix assumed to be orthonormal with det(R)=+1.

    Returns
    -------
    rvec : np.ndarray, shape (3,), dtype float32
        Axis-angle representation where direction is the rotation axis and the
        vector magnitude is the rotation angle in radians.
    """

    rvec, _ = cv2.Rodrigues(R)              # Use OpenCV to convert matrix form to Rodrigues vector
    return rvec.reshape(-1).astype(np.float32)  # Flatten to (3,) and cast to float32 for consistency

def project_to_SO3(R: np.ndarray) -> np.ndarray:
    """
    Project a rotation matrix to the nearest valid SO(3) matrix.
    """
    U, S, Vt = np.linalg.svd(R.astype(np.float64), full_matrices=False)     # SVD in higher precision
    R_hat = U @ Vt                              # Reconstruct rotation matrix                    
    if np.linalg.det(R_hat) < 0:                # Ensure a proper rotation (det=+1)
        U[:, -1] *= -1                          # Flip the sign of the last column of U
        R_hat = U @ Vt                          # Recompute R_hat with the corrected U
    return R_hat.astype(np.float32)             # Cast back to float32 for consistency

# =====================================
# Function: project_points_world
# =====================================

def project_points_world(R_WC, t_WC, K, dist, pts_W):
    """
    Project 3D world points into the image given a world-from-camera pose.

    Parameters
    ----------
    R_WC : np.ndarray, shape (3,3)
        Rotation from camera frame to world frame.
    t_WC : np.ndarray, shape (3,1)
        Camera origin expressed in world coordinates.
    K : np.ndarray, shape (3,3)
        Camera intrinsics matrix.
    dist : np.ndarray, shape (1,N) or (N,)
        Distortion coefficients.
    pts_W : np.ndarray, shape (N,3)
        3D points in world coordinates to be projected.

    Returns
    -------
    proj : np.ndarray, shape (N,2)
        2D pixel coordinates for each input 3D point.

    Implementation Detail
    ---------------------
    - cv2.projectPoints consumes Camera-from-World (R_CW, t_CW), so we invert
      (R_WC, t_WC) to build the required pose before projection.
    """
    R_CW = R_WC.T                        # Inverse rotation is transpose for orthonormal matrices
    t_CW = -R_CW @ t_WC                  # Invert translation according to rigid transform rules
    rvec = rodrigues_from_R(R_CW)        # Convert rotation matrix to Rodrigues vector for OpenCV
    proj, _ = cv2.projectPoints(         # Perform perspective projection with distortion
        pts_W.astype(np.float32),        # Ensure points are float32
        rvec,                            # Camera-from-World rotation in rvec form
        t_CW.astype(np.float32),         # Camera-from-World translation as float32
        K,                               # Intrinsic matrix
        dist                             # Distortion coefficients (can be zeros)
    )
    return proj.reshape(-1, 2)           # Return (N,2) array by removing singleton dimensions


# =====================================
# Class: MonoVO
# =====================================

class MonoVO:
    """
    Monocular Visual Odometry engine that composes pose in the camera frame.

    Coordinate Frames
    -----------------
    Internal: Camera-from-World (T_CW)
        X_cam = R_CW * X_world + t_CW
    Output/Visualization: World-from-Camera (T_WC)
        R_WC = R_CW^T
        t_WC = -R_CW^T * t_CW

    Responsibilities
    ----------------
    - Detect Shi–Tomasi corners and track with pyramidal LK.
    - Estimate Essential matrix (RANSAC) and recover relative (R, t).
    - Compose the internal pose (R_CW, t_CW) over time.
    - Cache points and inlier masks for visualization.
    - Maintain a world-frame trajectory history (t_WC columns).
    """

    def __init__(self, K: np.ndarray, dist: np.ndarray, max_features: int = 1200):
        self.K = K                               # Save intrinsics for all geometric operations
        self.dist = dist                         # Save distortion coefficients for projections
        self.max_features = max_features         # Max number of corners to track per frame

        self.prev_gray = None                    # Last grayscale frame (for optical flow)
        self.prev_pts = None                     # Last set of tracked feature points (N,1,2)

        # Internal camera-from-world pose state
        self.R_CW = np.eye(3, dtype=np.float32)  # Start with identity rotation
        self.t_CW = np.zeros((3, 1), np.float32) # Start at origin in camera coordinates

        # Trajectory storage and bookkeeping
        self.traj_cols = deque()                 # Queue of 3x1 columns for world positions
        self.good_count = 0                      # Number of valid tracked correspondences this frame

        # Visualization caches for current step
        self.last_prev_pts = None                # Previous (x,y) points used in the last motion estimate
        self.last_curr_pts = None                # Current (x,y) points used in the last motion estimate
        self.last_inliers  = None                # Boolean mask of inliers from recoverPose

    # ---------- utilities for frames ----------
    def get_pose_world(self):
        """
        Convert internal Camera-from-World (R_CW, t_CW) to World-from-Camera.

        Returns
        -------
        R_WC : np.ndarray, shape (3,3), float32
            Rotation that expresses camera axes in world coordinates.
        t_WC : np.ndarray, shape (3,1), float32
            Camera position in world coordinates.

        Notes
        -----
        R_WC = R_CW^T
        t_WC = -R_CW^T * t_CW
        """
        R_WC = self.R_CW.T                       # Invert rotation by transpose
        t_WC = -R_WC @ self.t_CW                 # Invert translation accordingly
        return R_WC.astype(np.float32), t_WC.astype(np.float32)  # Keep float32 consistency

    def get_traj_array(self) -> np.ndarray:
        """
        Assemble a (3xN) array of world-frame positions from stored columns.

        Returns
        -------
        traj : np.ndarray, shape (3,N), float32
            Sequence of camera positions in world coordinates (x,y,z as rows).

        Design
        ------
        Using a deque minimizes per-frame overhead; we concatenate only when asked.
        """
        if not self.traj_cols:                                   # No positions recorded yet
            return np.empty((3, 0), dtype=np.float32)            # Return an empty 3x0 array
        return np.concatenate(self.traj_cols, axis=1)            # Stack 3x1 columns into 3xN

    # ---------- feature detection ----------
    def _detect_corners(self, gray):
        """
        Detect Shi–Tomasi corners to track in a grayscale image.

        Parameters
        ----------
        gray : np.ndarray, shape (H,W), uint8
            Grayscale frame for corner detection.

        Returns
        -------
        pts : np.ndarray or None, shape (N,1,2)
            Subpixel-accurate points (x,y) or None if no corners found.

        Tuning Tips
        -----------
        - qualityLevel: relative min quality (0..1); higher → fewer, stronger corners.
        - minDistance: enforces spacing so corners aren’t clustered.
        - blockSize: neighborhood size for corner scoring.
        """
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_features,  # Upper bound on detected corners
            qualityLevel=0.01,                   # Corner quality threshold
            minDistance=7,                       # Enforce spacing between corners
            blockSize=7                          # Neighborhood size for detection
        )
        return pts                                # Return detected points or None

    # ---------- main step ----------
    def process(self, frame_bgr: np.ndarray):
        """
        Process a single BGR frame: track features, estimate motion, update pose.

        Parameters
        ----------
        frame_bgr : np.ndarray, shape (H,W,3), uint8
            Input color frame in BGR order (OpenCV default).

        Returns
        -------
        R_WC, t_WC : (np.ndarray, np.ndarray)
            Current pose expressed in world coordinates.

        Pipeline
        --------
        1) Convert to grayscale.
        2) Initialize if first frame (detect corners, log origin).
        3) Track prev→curr with pyramidal Lucas–Kanade.
        4) Filter valid correspondences; estimate Essential matrix with RANSAC.
        5) Recover relative pose (R_k, t_k); compose internal T_CW.
        6) Append current world position to trajectory.
        7) Prepare state for next iteration (prev_* buffers).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)   # Convert to grayscale for tracking

        if self.prev_gray is None:                           # First-frame setup path
            self.prev_gray = gray                            # Store as previous grayscale
            self.prev_pts = self._detect_corners(gray)       # Seed feature tracker
            _, t_WC0 = self.get_pose_world()                 # World position at start (origin)
            self.traj_cols.append(t_WC0.copy())              # Record the initial position
            self.last_prev_pts = None                        # Clear visualization caches
            self.last_curr_pts = None
            self.last_inliers  = None
            return self.get_pose_world()                     # Return current pose (identity)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(      # Track features to current frame
            self.prev_gray, gray, self.prev_pts, None,       # Provide prev frame and points
            winSize=(21, 21),                                # LK window size per pyramid level
            maxLevel=3,                                      # Number of pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)  # Early stop criteria
        )

        if next_pts is None or status is None:               # Complete tracking failure
            self.prev_gray = gray                            # Update previous frame regardless
            self.prev_pts = self._detect_corners(gray)       # Re-detect to attempt recovery
            self.last_prev_pts = None                        # Reset caches used for drawing
            self.last_curr_pts = None
            self.last_inliers  = None
            return self.get_pose_world()                     # Pose unchanged; return world pose

        status = status.squeeze().astype(bool)               # Convert (N,1) to (N,) bool mask
        prev_good = self.prev_pts[status]                    # Previous points with valid tracks
        curr_good = next_pts[status]                         # Current points matched to prev_good

        self.good_count = int(prev_good.shape[0])            # Count valid correspondences
        self.last_prev_pts = prev_good.reshape(-1, 2)        # Cache (M,2) for visual overlay
        self.last_curr_pts = curr_good.reshape(-1, 2)        # Cache (M,2) for visual overlay
        self.last_inliers  = None                            # Will be set after pose estimation

        if self.good_count >= 8:                             # Enough points to estimate Essential
            # Maintain correct temporal ordering: (prev, curr)
            E, _ = cv2.findEssentialMat(
                prev_good, curr_good, self.K,                # Use intrinsics for normalized coords
                method=cv2.RANSAC, prob=0.999, threshold=1.0 # Robust model with 1px error threshold
            )
            if E is not None:                                # Proceed only if a model is found
                _, R_k, t_k, mask_pose = cv2.recoverPose(    # Recover relative motion between frames
                    E, prev_good, curr_good, self.K          # Use same point order and intrinsics
                )
                # Normalize the rotation matrix to ensure it is a valid SO(3) matrix
                R_new = R_k @ self.R_CW
                self.R_CW = project_to_SO3(R_new)                          # Project to SO(3) if needed
                self.t_CW = (R_k @ self.t_CW + t_k).astype(np.float32)

                try:
                    self.last_inliers = mask_pose.squeeze().astype(bool)  # Save inlier mask for drawing
                except Exception:
                    self.last_inliers = None                               # Be robust to shape issues

                # Record the world position for trajectory visualization/exports
                _, t_WC = self.get_pose_world()                   # Convert pose to world frame
                self.traj_cols.append(t_WC.copy())                # Append as a new 3x1 column

        # Update previous-frame state for the next iteration
        self.prev_gray = gray                                     # Cache grayscale as previous
        if self.good_count < 0.3 * self.max_features:             # If too few points, refresh detection
            self.prev_pts = self._detect_corners(gray)            # Reseed feature tracker
        else:
            self.prev_pts = curr_good.reshape(-1, 1, 2)           # Keep tracking current good points

        return self.get_pose_world()                              # Return world pose to caller


# =====================================
# Function: plot_traj  (units-aware)
# =====================================

def plot_traj(arr: np.ndarray, units: str = "arb. units"):
    """
    Render a simple 3D plot of the trajectory with unit-aware axis labels.

    Parameters
    ----------
    arr : np.ndarray, shape (3,N)
        Stacked coordinate rows [x; y; z] over time.
    units : str
        Axis label suffix for clarity (e.g., "m" or "arb. units").

    Behavior
    --------
    - If matplotlib is not installed or trajectory is empty, the function logs
      a warning and returns without raising.
    """
    try:
        import matplotlib.pyplot as plt            # Lazy import to avoid hard dependency
        from mpl_toolkits.mplot3d import Axes3D    # noqa: F401 - ensure 3D plotting is registered
    except ImportError:
        print("[warn] matplotlib not installed → skipping plot")  # Inform the user gracefully
        return
    if arr.size == 0:                               # Nothing to visualize
        print("[warn] trajectory empty → skipping plot")
        return
    fig = plt.figure()                              # Create a new figure
    ax = fig.add_subplot(111, projection='3d')      # Add a 3D axes
    ax.plot(arr[0], arr[1], arr[2], marker='o', linewidth=1, markersize=2)  # Draw the path
    ax.set_xlabel(f'X [{units}]')                   # Label X with units
    ax.set_ylabel(f'Y [{units}]')                   # Label Y with units
    ax.set_zlabel(f'Z [{units}]')                   # Label Z with units
    ax.set_title(f'Monocular VO Trajectory ({units})')  # Provide context in title
    plt.tight_layout()                              # Reduce clipping of labels
    plt.show()                                      # Display the plot window


# =====================================
# Function: draw_overlay (units-aware HUD)
# =====================================

def draw_overlay(frame_bgr, vo: MonoVO, K, dist, idx,
                 scale_factor_for_axes=0.2, display_scale=1.0):
    """
    Draw a live visualization overlay on top of the input frame.

    Parameters
    ----------
    frame_bgr : np.ndarray, shape (H,W,3), uint8
        Original BGR frame to annotate.
    vo : MonoVO
        VO instance providing current pose and caches.
    K : np.ndarray, shape (3,3)
        Intrinsic matrix used for projecting axis endpoints.
    dist : np.ndarray
        Distortion vector applied during projection.
    idx : int
        Current frame index for the corner HUD.
    scale_factor_for_axes : float
        Length of the drawn camera axes (visual only).
    display_scale : float
        Multiplier for displaying position values in the HUD (e.g., meters).

    Returns
    -------
    vis : np.ndarray, shape (H,W,3), uint8
        Annotated frame with points, axes, and HUD text.

    Robustness
    ----------
    - Projection failures (e.g., behind camera, invalid pose) are caught; the
      function continues drawing available HUD elements.
    """
    vis = frame_bgr.copy()                         # Work on a copy to preserve the original input
    h, w = vis.shape[:2]                           # Image height and width for bounds/layout math

    # 1) Draw tracked feature points (green=inliers, red=outliers)
    if vo.last_curr_pts is not None:               # Only if we have points from the last step
        pts = vo.last_curr_pts.astype(np.float32)  # Ensure expected dtype
        if vo.last_inliers is not None and len(vo.last_inliers) == len(pts):
            inl = vo.last_inliers                  # Use the inlier mask from pose recovery
        else:
            inl = np.ones(len(pts), dtype=bool)    # If missing, treat all as inliers for display
        for p, ok in zip(pts, inl):                # Iterate point + inlier flag together
            x, y = int(p[0]), int(p[1])            # Cast to integer pixel coordinates
            if 0 <= x < w and 0 <= y < h:          # Draw only if the point lies inside the frame
                color = (50, 255, 50) if ok else (0, 0, 255)  # Green vs. red
                cv2.circle(vis, (x, y), 2, color, -1, lineType=cv2.LINE_AA)  # Small anti-aliased dot

    # 2) Draw camera axes and an origin marker (based on world-frame pose)
    try:
        R_WC, t_WC = vo.get_pose_world()           # Obtain world-frame pose for projection

        L = float(scale_factor_for_axes)           # Visual length of axes
        origin = t_WC.reshape(3, 1)                # Current camera origin in world coords as (3,1)
        x_end = origin + R_WC @ np.array([[L], [0.0], [0.0]], dtype=np.float32)  # X-axis endpoint
        y_end = origin + R_WC @ np.array([[0.0], [L], [0.0]], dtype=np.float32)  # Y-axis endpoint
        z_end = origin + R_WC @ np.array([[0.0], [0.0], [L]], dtype=np.float32)  # Z-axis endpoint

        pts_W = np.hstack([origin, x_end, y_end, z_end]).T  # (4,3) array of world points to project
        proj = project_points_world(R_WC, t_WC, K, dist, pts_W)  # Project to pixel coordinates

        c = tuple(map(int, proj[0]))               # Pixel for origin
        x = tuple(map(int, proj[1]))               # Pixel for X axis tip
        y = tuple(map(int, proj[2]))               # Pixel for Y axis tip
        z = tuple(map(int, proj[3]))               # Pixel for Z axis tip

        cv2.circle(vis, c, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)  # White origin marker
        cv2.putText(vis, "O", (c[0] + 6, c[1] - 6),                       # Label the origin
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(vis, c, x, (255, 0, 0), 2, lineType=cv2.LINE_AA)         # X axis (Blue in BGR)
        cv2.line(vis, c, y, (0, 255, 0), 2, lineType=cv2.LINE_AA)         # Y axis (Green)
        cv2.line(vis, c, z, (0, 0, 255), 2, lineType=cv2.LINE_AA)         # Z axis (Red)
        cv2.putText(vis, "X", x, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)  # 'X'
        cv2.putText(vis, "Y", y, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)  # 'Y'
        cv2.putText(vis, "Z", z, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)  # 'Z'
    except Exception:
        pass  # Continue rendering even if projection fails (e.g., degenerate geometry)

    # 3) Compose top-right HUD text (with display unit scaling if requested)
    s = float(display_scale)                        # Factor to scale printed coordinates (e.g., meters)
    _, t_WC = vo.get_pose_world()                   # Current camera position in the world frame
    t_disp = t_WC * s                               # Apply display scaling for the HUD
    hud_text = (f"good:{vo.good_count:4d} | "       # Number of tracked correspondences this frame
                f"x:{t_disp[0,0]:.4f} y:{t_disp[1,0]:.4f} z:{t_disp[2,0]:.4f}"
                f"{' m' if abs(s-1.0) > 1e-12 else ''}")  # Add 'm' if s==1 implies meters
    hud_font = cv2.FONT_HERSHEY_SIMPLEX             # Simple, readable font
    hud_scale = 0.6                                 # Font scaling
    hud_thick = 1                                   # Stroke thickness
    (tw, th), _ = cv2.getTextSize(hud_text, hud_font, hud_scale, hud_thick)  # Measure text box
    pad = 6                                         # Inner padding for the background box
    margin = 12                                     # Screen margin from the top-right corner
    x0 = w - tw - margin                            # X coordinate for box (right-aligned)
    y0 = margin                                     # Y coordinate near the top
    cv2.rectangle(vis, (x0 - 4, y0 - 4),            # Background box for legibility
                  (x0 + tw + pad, y0 + th + pad), (0, 0, 0), -1)
    cv2.putText(vis, hud_text, (x0, y0 + th),       # Render the HUD text in white
                hud_font, hud_scale, (255, 255, 255), hud_thick, cv2.LINE_AA)

    # 4) Bottom-right frame index indicator
    fr_text = f"{idx:04d}"                          # Zero-padded 4-digit index
    fr_font = cv2.FONT_HERSHEY_SIMPLEX              # Same font for consistency
    fr_scale = 0.5                                  # Slightly smaller than HUD
    fr_thick = 1                                    # Thin stroke
    (fw, fh), _ = cv2.getTextSize(fr_text, fr_font, fr_scale, fr_thick)  # Measure text
    margin = 10                                     # Margin from bottom-right corner
    org_x = w - margin - fw                         # Right-aligned x coordinate
    org_y = h - margin                              # Baseline y coordinate
    cv2.rectangle(vis, (org_x - 4, org_y - fh - 4), # Add a small background box behind the index
                  (org_x + fw + 4, org_y + 4), (0, 0, 0), -1)
    cv2.putText(vis, fr_text, (org_x, org_y),       # Draw the frame index
                fr_font, fr_scale, (255, 255, 255), fr_thick, cv2.LINE_AA)

    return vis                                      # Return the annotated image


# =====================================
# Function: main
# =====================================

def main():
    """
    Entry point: parse CLI options, run VO on the input, show/save results.

    CLI Options
    -----------
    --video : str or int
        Video file path or webcam index (e.g., 0).
    --intrinsics : str
        Path to .npz file containing arrays "K" and "dist".
    --output : str
        Path to save the trajectory as NumPy .npy (3xN).
    --csv : str (optional)
        If provided, also export the trajectory as CSV (Nx3).
    --plot : flag
        If set, show a 3D plot of the trajectory at the end.
    --frames : int
        Stop after processing N frames (0 means process all).
    --scale : float
        Constant factor to multiply translations when saving/plotting.
        (HUD uses display_scale; console uses this same factor)
    --no-show : flag
        Disable the live visualization window.
    --axis-len : float
        Visual length of axes drawn in the image (does not affect the VO state).

    Behavior
    --------
    - Opens the video source and runs the VO pipeline frame-by-frame.
    - Prints a console HUD each frame with scaled position.
    - If enabled, shows a live overlay with features, axes, and HUD.
    - On exit, saves trajectory (.npy and optional .csv) and optionally plots it.
    """
    ap = argparse.ArgumentParser(  # Build a user-friendly CLI
        description="Monocular Visual Odometry with absolute scale options + live visualization"
    )
    ap.add_argument("--video", required=True, help="Input video path (file/webcam index)")  # Source path or index
    ap.add_argument("--intrinsics", required=True, help=".npz file containing K and dist")  # Calibration archive
    ap.add_argument("--output", default="trajectory.npy", help="NumPy output file (3xN)")   # Output .npy path
    ap.add_argument("--csv", help="Optional CSV output path")                               # Optional CSV path
    ap.add_argument("--plot", action="store_true", help="Show 3D plot")                     # Enable plotting
    ap.add_argument("--frames", type=int, default=0, help="Process at most N frames (0 = all)")  # Max frames
    ap.add_argument("--scale", type=float, default=None,                                     # Absolute scale factor
                    help="Multiply all translations by constant scale (meters/arbitrary units)")
    ap.add_argument("--no-show", action="store_true", help="Do not open live visualization window")  # Headless mode
    ap.add_argument("--axis-len", type=float, default=0.2,                                   # Axis length for overlay
                    help="Axis length in world units for on-frame visualization")
    args = ap.parse_args()                                                                   # Parse the CLI args

    # Open the capture device or file
    try:
        vid = int(args.video)                    # Try to interpret --video as webcam index
        cap = cv2.VideoCapture(vid)              # Open default camera device by index
    except ValueError:
        cap = cv2.VideoCapture(args.video)       # Otherwise treat it as a file path

    if not cap.isOpened():                       # Validate that the source opened correctly
        sys.exit(f"[error] Cannot open video: {args.video}")  # Exit with an error message

    # Load calibration and initialize VO engine
    K, dist = load_intrinsics(Path(args.intrinsics))  # Read K and dist from .npz file
    vo = MonoVO(K, dist)                              # Create the VO pipeline instance

    window_name = "Monocular VO (Live)"               # Title for OpenCV window
    if not args.no_show:                              # Only create a window if visualization is on
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)  # Auto-sized window based on frame size

    idx = 0                                           # Frame counter
    t0 = time.time()                                  # Start timing for FPS calculation
    while True:                                       # Main processing loop
        if args.frames and idx >= args.frames:        # Stop if a maximum frame count is specified
            break
        ok, frame = cap.read()                        # Grab the next frame from the source
        if not ok:                                    # Break on EOF or read error
            break

        vo.process(frame)                             # Run the VO step for this frame

        # Console HUD (units-aware)
        s = args.scale if args.scale is not None else 1.0  # Scale factor for printed position
        _, t_WC = vo.get_pose_world()                # Query current world pose
        t_disp = t_WC * s                            # Apply user-specified scaling
        print(f"{idx:04d} | good:{vo.good_count:4d} | "   # Frame index + good correspondence count
              f"x:{t_disp[0,0]:.4f} y:{t_disp[1,0]:.4f} z:{t_disp[2,0]:.4f}"
              f"{' m' if abs(s-1.0)>1e-12 else ''}")     # Append 'm' if the scale implies meters

        # Live overlay window (if enabled)
        if not args.no_show:                         # Only if user didn’t request headless mode
            vis = draw_overlay(                      # Generate annotated frame
                frame, vo, K, dist, idx,
                scale_factor_for_axes=args.axis_len, # Axis length in the overlay
                display_scale=(args.scale if args.scale is not None else 1.0)  # HUD scale
            )
            cv2.imshow(window_name, vis)             # Display the overlay
            key = cv2.waitKey(1) & 0xFF              # Non-blocking key check
            if key == ord('q'):                      # Quit immediately
                break
            if key == ord('p'):                      # Pause the stream
                while True:                          # Stay paused until 'p' or 'q'
                    k2 = cv2.waitKey(30) & 0xFF
                    if k2 in (ord('p'), ord('q')):
                        if k2 == ord('q'):           # Quit from pause as well
                            cap.release()
                            cv2.destroyAllWindows()
                            elapsed = time.time() - t0
                            fps = (idx/elapsed) if elapsed > 0 else 0.0
                            print(f"[info] Finished {idx} frames in {elapsed:.2f}s (≈{fps:.1f} FPS)")
                            return
                        break                        # Resume on 'p'

        idx += 1                                     # Increment the frame counter

    cap.release()                                    # Release file/camera handle
    cv2.destroyAllWindows()                          # Close any OpenCV windows that were opened

    elapsed = time.time() - t0                       # Total elapsed processing time
    fps = (idx/elapsed) if elapsed > 0 else 0.0      # Average frames per second
    print(f"[info] Finished {idx} frames in {elapsed:.2f}s (≈{fps:.1f} FPS)")  # Summary line

    # Retrieve accumulated trajectory in world frame
    traj = vo.get_traj_array()                       # (3xN) array with columns = positions over time
    if traj.size == 0:                               # Nothing recorded (e.g., very short/empty video)
        print("[warn] Empty trajectory → skipping save/plot")  # Inform user and exit gracefully
        return

    # Apply absolute scale if requested (for saving/plotting only)
    scale_value = 1.0                                # Default: arbitrary units
    if args.scale is not None:                       # User provided a scale factor
        scale_value = float(args.scale)              # Parse/normalize to float
        traj = traj * scale_value                    # Scale entire trajectory
        print(f"[info] Applied constant scale = {scale_value:g}")  # Log the applied scale
    else:
        print("[info] No scale adjustment applied (arbitrary units)")  # Clarify that units are arbitrary

    # Save the trajectory as .npy for precise reuse
    np.save(args.output, traj.astype(np.float32))    # Persist as float32 to reduce size and align with OpenCV
    print(f"[info] Saved NumPy → {args.output}  shape={traj.shape}")  # Confirm output path and shape

    # Optionally also save as CSV for quick inspection in spreadsheets
    if args.csv:                                     # Only if a CSV path was provided
        header = "x(m),y(m),z(m)" if abs(scale_value-1.0) > 1e-12 else "x,y,z"  # Unit-aware header
        np.savetxt(args.csv, traj.T, delimiter=",", header=header, comments="")  # Transpose to N x 3
        print(f"[info] Saved CSV   → {args.csv}")    # Confirm CSV path

    # Optionally show a simple 3D plot of the path
    if args.plot:                                    # If plotting was requested
        units = "m" if abs(scale_value-1.0) > 1e-12 else "arb. units"  # Choose label units
        plot_traj(traj, units)                       # Render the trajectory plot


if __name__ == "__main__":  # Run only if this file is executed as a script
    main()                   # Execute the main entry point