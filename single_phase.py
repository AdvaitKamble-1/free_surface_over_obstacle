# Data Plotting and Extraction - Single-Phase

# Author: Advait Kamble
# Date: 27/06/2026

import numpy as np
import matplotlib.pyplot as plt
import os

# Defining the parameters and filename

filename = "00001_single_phase.txt"
t_start = 100.0   # discard transient before this time
D = 1.0 # cylinder diameter
U = 1.0 # free-stream velocity


# Toggles to improve workflow: True -> to print, show or save. False ->
# to not do that.

PRINT_BLOCKS = {
    "data_info" : True,
    "stats"     : True,
}

SHOW_PLOTS = {
    "full_series"   : False,
    "steady_series" : True,
}

SAVE_PLOTS = {
    "full_series"   : True,
    "steady_series" : True,
}

# Helper functions to save and show plots.
def maybe_save(key, fname, dpi=150):
    if SAVE_PLOTS[key]:
        plt.savefig(fname, dpi=dpi)
def maybe_show(key):
    if SHOW_PLOTS[key]:
        plt.show()
    else:
        plt.close()

# Function written by assistance of Claude to extract
# data from the text file

def parse_file(filepath):
    times, Cls, Cds = [], [], []
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return np.array(times), np.array(Cls), np.array(Cds)
    with open(filepath, 'r') as f:
        current_time = None
        current_Cl = None
        current_Cd = None
        for line in f:
            stripped = line.strip()
            if 'ns =' in stripped and 'time =' in stripped:
                if current_time is not None and current_Cl is not None and current_Cd is not None:
                    times.append(current_time)
                    Cls.append(current_Cl)
                    Cds.append(current_Cd)
                current_time = float(stripped.split('time =')[1].strip())
                current_Cl = None
                current_Cd = None
            elif stripped.startswith('Cd'):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        current_Cd = float(parts[-1])
                    except ValueError:
                        current_Cd = None
            elif stripped.startswith('Cl'):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        current_Cl = float(parts[-1])
                    except ValueError:
                        current_Cl = None
        if current_time is not None and current_Cl is not None and current_Cd is not None:
            times.append(current_time)
            Cls.append(current_Cl)
            Cds.append(current_Cd)
    return np.array(times), np.array(Cls), np.array(Cds)

# Filter the data and apply masks (initial spike gets masked)

times, Cls, Cds = parse_file(filename)
if PRINT_BLOCKS["data_info"]:
    print(f"Records read: {len(times)}")
    print(f"Time range: {times[0]:.4f} — {times[-1]:.4f} s")
mask = times > t_start
times_s = times[mask]
Cls_s = Cls[mask]
Cds_s = -Cds[mask] # positive drag -> matches sign convention in studies

# Thresholds to ensure only valid values are selected
valid = ~(np.isnan(Cls_s) | np.isnan(Cds_s)) & (np.abs(Cls_s) < 1e4) & (np.abs(Cds_s) < 1e4)
times_s = times_s[valid]
Cls_s = Cls_s[valid]
Cds_s = Cds_s[valid]

# Plotting the full time-series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(times, -Cds, linewidth=0.6, color='red')
ax1.axvline(t_start, color='black', lw=0.8, ls='--', label=f't_start = {t_start} s')
ax1.set_ylabel(r'$C_d$')
ax1.set_title('Single Phase — Drag and Lift Coefficients')
ax1.grid(True, alpha=0.4)
ax1.legend(fontsize=8)
ax2.plot(times, Cls, linewidth=0.6, color='blue')
ax2.axvline(t_start, color='black', lw=0.8, ls='--')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel(r'$C_l$')
ax2.grid(True, alpha=0.4)
plt.tight_layout()
maybe_save("full_series", "single_phase_full.png")
maybe_show("full_series")

# Plotting the steady state time-series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(times_s, Cds_s, linewidth=0.6, color='red')
ax1.set_ylabel(r'$C_d$')
ax1.set_title(f'Single Phase — Steady State (t > {t_start} s)')
ax1.grid(True, alpha=0.4)
ax2.plot(times_s, Cls_s, linewidth=0.6, color='blue')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel(r'$C_l$')
ax2.grid(True, alpha=0.4)
plt.tight_layout()
maybe_save("steady_series", "single_phase_steady.png")
maybe_show("steady_series")

# Printing useful stats -> mean drag, mean lift, strouhal number
mean_Cd = np.mean(Cds_s)
mean_Cl = np.mean(Cls_s)
Clrms = np.sqrt(np.mean(Cls_s**2))

dt = np.median(np.diff(times_s))
fft_mag = np.abs(np.fft.rfft(Cls_s - mean_Cl))
freqs = np.fft.rfftfreq(len(Cls_s), d=dt)
fft_mag[0] = 0 # ignore DC
f_peak = freqs[np.argmax(fft_mag)]
St = f_peak*D/U

# Summary print block -> written using Claude
if PRINT_BLOCKS["stats"]:
    print(f"\nSteady-state stats (t > {t_start} s)  —  {len(times_s)} samples")
    print("=" * 50)
    print(f"  Mean Cd  : {mean_Cd:.6f}")
    print(f"  Mean Cl  : {mean_Cl:.6f}")
    print(f"  Cl RMS   : {Clrms:.6f}")
    print(f"  f_shed   : {f_peak:.6f} Hz")
    print(f"  St       : {St:.6f}   (St = f·D/U, D={D}, U={U})")
    print("=" * 50)
