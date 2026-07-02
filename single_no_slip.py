# Single-Phase Poiseuille flow velocity profile

# Author: Advait Kamble
# Date: 27/06/2026

import numpy as np
import matplotlib.pyplot as plt
import os

# Computational Domain Parameters
nx = 768; ny = 256
lx = 38.4; ly = 12.8

mu = 0.005556

binary_files = [
    ("u_00050.bin", 5.0),
    # ("u_00250.bin", 25.0),
    # ("u_01000.bin", 100.0),
    ("u_01911.bin", 191.1)
]

x_target = 19.2 # Half the channel length
dx = lx/nx
dy = ly/ny
y_phys = (np.arange(ny + 2) - 0.5)*dy
i_col = round(x_target/dx + 0.5)
# Infer dp/dx from the steady-state profile (last entry)
steady_file = binary_files[-1][0]
u_steady = np.fromfile(steady_file, dtype=np.float64).reshape((ny + 2, nx + 2))[:, i_col]
h = ly/2 # channel half-width
d2u_dy2 = np.gradient(np.gradient(u_steady[1:-1], dy), dy)
dpdx = mu * np.mean(d2u_dy2)

# Analytical Poiseuille solution using centred form, shifted to physical y
y_ana_c = np.linspace(-h, h, 500)
u_ana = (-dpdx)*(h**2/(2*mu))*(1 - y_ana_c**2/h**2)
y_ana = y_ana_c + h
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Numerical
for binary_file, dump_time in binary_files:
    u_field = np.fromfile(binary_file, dtype=np.float64).reshape((ny + 2, nx + 2))
    u_profile = u_field[:, i_col]
    ax1.plot(u_profile, y_phys, 'o-', markersize=2, label=f"t = {dump_time}s")

ax1.set_xlabel("u", fontsize=16)
ax1.set_ylabel("y", fontsize=16)
ax1.set_title("(a)", fontsize=18)
ax1.legend(fontsize=14)
ax1.tick_params(labelsize=13)
ax1.grid(True)
ax1.set_box_aspect(1)

# Analytical
ax2.plot(u_ana, y_ana, 'k-', label="Analytical")
ax2.set_xlabel("u", fontsize=16)
ax2.set_ylabel("y", fontsize=16)
ax2.set_title("(b)", fontsize=18)
ax2.legend(fontsize=14)
ax2.tick_params(labelsize=13)
ax2.grid(True)
ax2.set_box_aspect(1)

plt.tight_layout()
plt.show()

# print(dpdx)

# NOTE: The analytical solution is derived from the Poiseuille flow equation for a channel with no-slip boundary conditions. 
#       The velocity profile is parabolic, and the maximum velocity occurs at the center of the channel. The numerical 
#       results are compared against this analytical solution to validate the simulation.
# NOTE: AI (Claude) was used to assist in the development of this code, particularly in structuring the analytical solution 
#       and plotting routines whose references are provided in the report.

