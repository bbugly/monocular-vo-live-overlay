import numpy as np

# Load the existing npz file (contains camera intrinsics and distortion coefficients)
data = np.load("intrinsics.npz")

# Copy the intrinsic matrix K from the loaded file
K = data['K'].copy()

# Update intrinsic parameters (fx, fy, cx, cy)
K[0, 0] = 914.01   # fx: focal length in pixels along x-axis
                   #     fx = (F * px) / sx
                   #     F = focal length in mm
                   #     px = image width in pixels
                   #     sx = sensor width in mm

K[1, 1] = 541.13   # fy: focal length in pixels along y-axis
                   #     fy = (F * py) / sy
                   #     F = focal length in mm
                   #     py = image height in pixels
                   #     sy = sensor height in mm

K[0, 2] = 640.00   # cx: principal point x-coordinate (usually image_width / 2)
K[1, 2] = 360.00   # cy: principal point y-coordinate (usually image_height / 2)

# Define distortion coefficients (assuming no lens distortion)
dist = np.array([0., 0., 0., 0., 0.], dtype=np.float32)

# Save the updated intrinsic parameters and distortion coefficients back to npz file
np.savez("intrinsics.npz", K=K, dist=dist)

# Print updated values to verify
print("K =\n", K)
print("dist =", dist)