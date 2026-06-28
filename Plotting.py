# Plotting script for velocity contours

# Author: Advait Kamble
# Date: 27/06/2026

# Note: use of AI (CoPilot, Claude) to assist with writing conditional operations, 
#       polishing and improving presentability of contour plots.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# "full" -> full-domain velocity contour saved to contour_file/
# "closeup"-> near-wake velocity + VOF side by side, saved to wake_dir/
# "farwake" -> far-downstream velocity + VOF side by side, saved to farwake_dir/
VELOCITY_MODE = "full"

data = "tester"
contour_file = f"{data}_contours"

data_dir = Path(__file__).resolve().parent / data
quiver_dir = Path(__file__).resolve().parent / contour_file

# Phase toggle
SINGLE_PHASE = False

if SINGLE_PHASE:
    nx_int = 768
    ny_int = 256
    lx = 38.4
    ly = 12.8
else:
    nx_int = 2048
    ny_int = 256
    lx = 64.0
    ly = 8.0

# Full array size including ghost cells
nx = nx_int + 2
ny = ny_int + 2
dx = lx/nx_int
dy = ly/ny_int
dtype = np.float64

obstacle_file = "alpha_00001.bin"
def load_field(path):
    a = np.fromfile(path, dtype=dtype)
    return a.reshape((ny, nx))

# Physical coordinates at cell centres
x = (np.arange(nx) - 0.5)*dx
y = (np.arange(ny) - 0.5)*dy
X, Y = np.meshgrid(x, y)

lb = 1.0
x_center = lx*0.2
y_center = ly * 0.5125 + 0.5*lb

# Near-wake region: 2 diameters upstream to 15 downstream, ±3 diameters vertically
x_wake_lo = x_center - 2.0*lb
x_wake_hi = x_center + 15.0*lb
y_wake_lo = y_center - 3.0*lb
y_wake_hi = y_center + 3.0*lb

i_lo = int(np.searchsorted(x, x_wake_lo))
i_hi = int(np.searchsorted(x, x_wake_hi))
j_lo = int(np.searchsorted(y, y_wake_lo))
j_hi = int(np.searchsorted(y, y_wake_hi))

# Far-downstream region: 25–45 diameters downstream, ±3 diameters vertically
x_far_lo = x_center + 25.0*lb
x_far_hi = x_center + 45.0*lb
i_far_lo = int(np.searchsorted(x, x_far_lo))
i_far_hi = int(np.searchsorted(x, x_far_hi))

# Load obstacle
obstacle_path = data_dir/obstacle_file
obstacle = load_field(obstacle_path)
obstacle_true = obstacle > 0.5

# Centroid of the obstacle from the actual field data
j_obs, i_obs = np.where(obstacle_true)
x_obs_center = float(np.mean(x[i_obs]))
y_obs_center = float(np.mean(y[j_obs]))

# Recompute vertical wake bounds using the true centroid
j_lo = int(np.searchsorted(y, y_obs_center - 3.0*lb))
j_hi = int(np.searchsorted(y, y_obs_center + 3.0*lb))

out_obstacle_dir = Path(__file__).resolve().parent/"contours_alpha"
out_obstacle_dir.mkdir(exist_ok=True)

wake_dir = Path(__file__).resolve().parent/f"{data}_wake"
wake_dir.mkdir(exist_ok=True)

farwake_dir = Path(__file__).resolve().parent/f"{data}_farwake"
farwake_dir.mkdir(exist_ok=True)

# Obstacle-only plot
plt.figure(figsize=(10, 5))
cf = plt.contourf(X, Y, obstacle, levels=100, cmap="viridis")
plt.colorbar(cf)
plt.contour(X, Y, obstacle_true.astype(float), levels=[0.5], colors="black", linewidths=2)
plt.title("Obstacle Contour")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal")
plt.legend()
plt.tight_layout()
plt.savefig(out_obstacle_dir / "obstacle_contour.png", dpi=300)
#plt.show()
plt.close()
# Extract velocity files
v_vels_all = sorted(data_dir.glob("v_*.bin"))
u_vels_all = sorted(data_dir.glob("u_*.bin"))

stride = 200
v_vels = v_vels_all#[::stride]
u_vels = u_vels_all#[::stride]

for u_file, v_file in zip(u_vels, v_vels):
    u = load_field(u_file)
    v = load_field(v_file)
    u_masked = np.ma.masked_where(obstacle_true, u)
    v_masked = np.ma.masked_where(obstacle_true, v)
    C = np.ma.sqrt(u_masked**2 + v_masked**2)
    timestep = u_file.name.split("_")[1]
    vof_file = data_dir/f"vof_{timestep}"
    vof = load_field(vof_file)
    C_liquid = np.ma.masked_where(vof < 0.5, C)
    if VELOCITY_MODE == "full":
        # Full-domain velocity contour (separate figure)
        plt.figure(figsize=(12, 5))
        cf = plt.contourf(X, Y, C_liquid, levels=50, cmap="viridis")
        plt.colorbar(cf, label="|u|")
        plt.contour(X, Y, obstacle_true.astype(float), levels=[0.5], colors="black", linewidths=1.5)
        plt.contour(X, Y, vof, levels=[0.5], colors="black", linewidths=1.5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Velocity Magnitude Contour ({timestep})")
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        save_dir = Path(__file__).resolve().parent/contour_file
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir/f"velocity_contour_{timestep}.png", dpi=300)
        print(f"Saved full contour plot for {timestep}")
        plt.close()

    elif VELOCITY_MODE == "closeup":  # near-wake velocity and VOF side by side
        Xw = X[j_lo:j_hi, i_lo:i_hi]
        Yw = Y[j_lo:j_hi, i_lo:i_hi]
        obs_w = obstacle_true[j_lo:j_hi, i_lo:i_hi]
        vof_w = vof[j_lo:j_hi, i_lo:i_hi]
        C_liq_w = C_liquid[j_lo:j_hi, i_lo:i_hi]
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
        cf = ax1.contourf(Xw - x_obs_center, Yw - y_obs_center, C_liq_w, levels=50, cmap="viridis")
        plt.colorbar(cf, ax=ax1, label="|u|")
        ax1.contour(Xw - x_obs_center, Yw - y_obs_center, obs_w.astype(float), levels=[0.5], colors="black", linewidths=1.5)
        ax1.set_xlabel("x/lb")
        ax1.set_ylabel("y/lb")
        ax1.set_title(f"Near-Wake Velocity ({timestep})")
        ax1.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(wake_dir/f"wake_contour_{timestep}.png", dpi=300)
        print(f"Saved close-up contour plot for {timestep}")
        plt.close()

    else:  # "farwake" — far-downstream velocity and VOF side by side
        Xf = X[j_lo:j_hi, i_far_lo:i_far_hi]
        Yf = Y[j_lo:j_hi, i_far_lo:i_far_hi]
        obs_f = obstacle_true[j_lo:j_hi, i_far_lo:i_far_hi]
        vof_f = vof[j_lo:j_hi, i_far_lo:i_far_hi]
        C_liq_f = C_liquid[j_lo:j_hi, i_far_lo:i_far_hi]
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
        cf = ax1.contourf(Xf - x_obs_center, Yf - y_obs_center, C_liq_f, levels=50, cmap="viridis")
        plt.colorbar(cf, ax=ax1, label="|u|")
        ax1.contour(Xf - x_obs_center, Yf - y_obs_center, obs_f.astype(float), levels=[0.5], colors="black", linewidths=1.5)
        ax1.set_xlabel("x/lb")
        ax1.set_ylabel("y/lb")
        ax1.set_title(f"Far-Wake Velocity ({timestep})")
        ax1.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(farwake_dir/f"farwake_contour_{timestep}.png", dpi=300)
        print(f"Saved far-wake contour plot for {timestep}")
        plt.close()

print("All contour plots have been saved")
