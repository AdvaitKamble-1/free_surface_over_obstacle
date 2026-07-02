# Grid Resolution Study and Experimental Hydrodynamics Data Comparison

# Author: Advait Kamble
# Date: 27/06/2026

import numpy as np
import matplotlib.pyplot as plt
import os

files = {
    "coarse": "00001_coarse.txt",
    "medium": "00001_medium.txt",
    "fine": "00001_fine.txt"
}

# File Format:

# ns =        1      time =     0.0000010
#              Cd    2.095928942741307E+04
#              Cl   -5.239822356853269E-01
# VS Transverse P   -1.389869433477288E-04
# DA Streamwise P    1.242876945796269E+00
# DA Transverse P    1.216645915816748E-02

def parse_file(filename):
    """
    Parse the given text file and extract the relevant data into a dictionary.
    """
    ns, times, Cd, Cl, VS_trans = [], [], [], [], []
    with open(filename, 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith("ns ="):
                ns.append(int(s.split("ns =")[1].split()[0]))
                times.append(float(s.split("time =")[1].strip()))
            elif s.startswith("Cd"):
                Cd.append(float(s.split("Cd")[1].strip()))
            elif s.startswith("Cl"):
                Cl.append(float(s.split("Cl")[1].strip()))
            elif s.startswith("VS Transverse P"):
                VS_trans.append(float(s.split("VS Transverse P")[1].strip()))
    return {
        "ns": np.array(ns),
        "times": np.array(times),
        "Cd": np.array(Cd),
        "Cl": np.array(Cl),
        "VS_trans": np.array(VS_trans),
    }
data = {label: parse_file(filename) for label, filename in files.items()}
t_start = 5.0
D = 1.0  # cylinder diameter
U = 1.0  # free-stream velocity
masks = {label: d["times"] >= t_start for label, d in data.items()}
for label, d in data.items():
    mask = masks[label]
    Cl_masked = d["Cl"][mask]
    times_masked = d["times"][mask]
    mean_Cd = np.mean(d["Cd"][mask])
    mean_Cl = np.mean(Cl_masked)
    if len(Cl_masked) >= 4:
        dt_cl = np.median(np.diff(times_masked))
        fft_mag = np.abs(np.fft.rfft(Cl_masked - mean_Cl))
        freqs = np.fft.rfftfreq(len(Cl_masked), d=dt_cl)
        fft_mag[0] = 0
        St = freqs[np.argmax(fft_mag)]*D/U
    else:
        St = float("nan")

    print(f"[{label}] Mean Cd = {mean_Cd:.6f}  |  Mean Cl = {mean_Cl:.6f}  |  St = {St:.6f}")

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, key, ylabel in zip(axes, ("Cd", "Cl", "VS_trans"),
                            ("Drag Coefficient $C_d$", "Lift Coefficient $C_l$", "VS Transverse Probe Velocity")):
    for label, d in data.items():
        mask = masks[label]
        ax.plot(d["times"][mask], d[key][mask], linewidth=0.6, label=label)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True, linewidth=0.4)
    ax.tick_params(axis='both', labelsize=16)
axes[-1].set_xlabel("Time (s)", fontsize=18)
fig.tight_layout()
out_path = "grid_resolution_comparison.png"
fig.savefig(out_path, dpi=150)

# Corresponds to the experimental configuration with chosen grid resolution
medium = data["medium"]
mask = masks["medium"]
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(medium["times"][mask], medium["Cd"][mask], linewidth=1.5, label="$C_d$")
ax2.plot(medium["times"][mask], medium["Cl"][mask], linewidth=1.5, label="$C_l$")
ax2.set_xlabel("Time (s)", fontsize=18)
ax2.set_ylabel("Coefficient", fontsize=18)
ax2.legend(fontsize=16, title="Experimental Configuration", title_fontsize=16)
ax2.grid(True, linewidth=0.4)
ax2.tick_params(axis='both', labelsize=16)
fig2.tight_layout()
out_path2 = "medium_experimental_configuration.png"
fig2.savefig(out_path2, dpi=150)
print(f"Saved {out_path2}")

plt.show()
print(f"Saved {out_path}")

# NOTE: AI (Claude) was used to assist in generating this code, but the final implementation and validation were performed by the author.
#       This code will only work if the input files are present in the same directory as this script.

