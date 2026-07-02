# Local Froude number profile - FDM results

# Author: Advait Kamble
# Date: 27/06/2026

import numpy as np
import matplotlib.pyplot as plt

# Initial parameters
eta_file = "/home/advait/CFD/test/text_code/eta_history.bin"
nx = 2048
lx = 64.0
x_cyl = lx*0.2
dt_output = 0.1
U = 1.0
g = 25.0
froude_times = [5.0, 25.0]

# Load Data
rec_size = nx + 3
raw = np.fromfile(eta_file, dtype=np.float64)
n_steps = raw.size//rec_size
data = raw[:n_steps*rec_size].reshape(n_steps, rec_size)
time_array = data[:, 0]
eta_array = data[:, 2:-1]
# Compute local Froude number (initial Froude number Fr0 = U/sqrt(g*h0))
dx = lx/nx
x = (np.arange(nx) + 0.5)*dx
h0 = np.mean(eta_array[0, :])
Fr0 = U/np.sqrt(g*h0)

# Plot local Froude number profiles at specified times
fig, ax = plt.subplots(figsize=(12, 5))
for t_target in froude_times:
    idx = np.argmin(np.abs(time_array - t_target))
    t_actual = time_array[idx]
    Fr = U / np.sqrt(g*eta_array[idx, :])
    ax.plot(x, Fr, lw=1.5, label=f"t = {t_actual:.1f} s")
ax.axvline(x_cyl, color="black", lw=1, ls="--", label=f"Cylinder x = {x_cyl:.1f} m")
ax.axhline(Fr0,   color="gray",  lw=0.8, ls=":", label=f"Background Fr₀ = {Fr0:.4f}")
ax.set_xlabel("Streamwise position x (m)", fontsize=16)
ax.set_ylabel("Local Froude number Fr(x)", fontsize=16)
ax.set_title(f"FDM — Local Froude Number Profile  (Fr₀ = {Fr0:.4f})", fontsize=18)
ax.tick_params(labelsize=14)
ax.set_xlim(0, lx)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# NOTE: AI (Claude) was used to assist in writing this code. The AI provided suggestions and code snippets
#       , which were then reviewed.
