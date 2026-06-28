# Code to plot amplitude contour

# Author: Advait Kamble
# Date: 27/06/2026

import numpy as np
import matplotlib.pyplot as plt
import os

OUTDIR = "amplitude_contours"
os.makedirs(OUTDIR, exist_ok=True)

# Current amplitude simulation stream-wise parameters - resolution, length
baseline = {
    "file": "eta_history.bin",
    "nx"  : 2048,
    "lx"  : 64.0,
}

# Use of Claude to assist with writing this function
def load_eta_history(file, nx, lx):
    rec_size = nx + 3  # time, step index, nx eta values (interface at each spatial location)
    raw = np.fromfile(file, dtype=np.float64)
    n_steps = raw.size//rec_size
    data = raw[:n_steps*rec_size].reshape(n_steps, rec_size) # Drop any trailing partial record
    time_array = data[:, 0]
    eta_array = data[:, 2:-1] # Skips time/step columns and trailing column
    dx = lx/nx
    x = (np.arange(nx) + 0.5)*dx # Cell-centered grid -> Same as solver
    eta_ref = eta_array[0, :] # Initial interface shape, used as the reference for amplitude
    amplitude_array = eta_array - eta_ref[np.newaxis, :]
    return x, time_array, amplitude_array

# Use of Claude to assist with writing this function
def plot_amplitude_contour(x, time_array, amplitude_array, x_cyl, title, outfile, show_cylinder_line=True, xlim=None, ylim=None):
    X, T = np.meshgrid(x, time_array)
    fig, ax = plt.subplots(figsize=(12, 6))
    cf = ax.contourf(X, T, amplitude_array, levels=100, cmap="RdBu_r")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Interface amplitude (m)", fontsize=15)
    cbar.ax.tick_params(labelsize=14)
    if show_cylinder_line:
        ax.axvline(x_cyl, color="black", lw=2.5, ls="--", label=f"Cylinder x={x_cyl:.1f} m")
        ax.legend(fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("x (m)", fontsize=15)
    ax.set_ylabel("Time (s)", fontsize=15)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)

x, time_array, amplitude_array = load_eta_history(baseline["file"], baseline["nx"], baseline["lx"])
x_cyl = 0.2*baseline["lx"]

# Full baseline contour
plot_amplitude_contour(
    x, time_array, amplitude_array, x_cyl,
    title="Interface Amplitude Contour (baseline)",
    outfile=os.path.join(OUTDIR, "amplitude_contour_baseline.png"),
)

# Zoomed-in snippet of the baseline contour
SNIPPET_XLIM = (0.0, 64.0) # (x_min, x_max) in m
SNIPPET_YLIM = (0.0, 10.0) # (t_min, t_max) in s

plot_amplitude_contour(
    x, time_array, amplitude_array, x_cyl,
    title="Interface Amplitude Contour (baseline, snippet)",
    outfile=os.path.join(OUTDIR, "amplitude_contour_baseline_snippet.png"),
    xlim=SNIPPET_XLIM,
    ylim=SNIPPET_YLIM,
)
