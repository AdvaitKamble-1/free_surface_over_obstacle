"""
Wave Tracking Algorithm.
"""

# Author: Advait Kamble
# Date: 26/06/2026

import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "wave_tracking_results"
os.makedirs(OUTDIR, exist_ok=True)

# Loading the binary file

eta_file = "eta_history.bin"
nx = 2048
lx = 64.0
dt_output = 0.1

# Linear dispersion relation (deep/finite-depth gravity wave speed relative
# to background flow U), used as the theoretical reference for the tracked
# wave speeds.
U = 1.0
g = 25.0
D = 1.0

k = D / (2 * np.pi)
c_wave = np.sqrt(g * k)
c_plus = U + c_wave # downstream-travelling wave speed
c_minus = U - c_wave # upstream-travelling wave speed

# (name, x_min, x_max, thresh_factor, c_min_dn, c_max_dn, c_min_up, c_max_up,
#  min_track_len, min_dx_dn, min_dx_up)
# Each region requires a different threshold factor because of the presence
# of different wave-structures in each region. 

REGIONS = [
    ("upstream", 5.0, 12.0, 0.01, c_plus - 0.5, c_plus + 0.5, -1.0, -0.05, 3, 1.0, 0.2),
    ("near_wake", 13.0, 35.0, 0.3, 0.8, 1.5, -1.0, -0.05, 3, 1.0, 0.0),
    ("downstream", 35.0, lx, 0.2, 0.8, 1.5, -1.0, -0.05, 3, 1.0, 0.2),
]

# Re-structuring the data
# nx + 3 accounts for two values at either end and the time column
rec_size = nx + 3
raw = np.fromfile(eta_file, dtype=np.float64)
n_steps = raw.size//rec_size
data = raw[:n_steps*rec_size].reshape(n_steps, rec_size)
time_array = data[:, 0]
eta_array = data[:, 2:-1]

dx = lx/nx
x  = (np.arange(nx) + 0.5)*dx # Cell-centred grid

# eta_ref is the interface profile at t = 0 (undisturbed)
eta_ref = eta_array[0, :]
amplitude_array = eta_array - eta_ref[np.newaxis, :]

# Algorithm relevant functions
def detect_peaks(amp, x_loc, time_arr, threshold):
    """Find local extrema in abs(amplitude) above threshold at each timestep.
    """
    t_list, x_list = [], []
    for i in range(amp.shape[0]):
        row = amp[i, :]
        for j in range(1, len(row) - 1):
            # Peaks correspond to either the crests or troughs, so either is
            # a peak.
            if (abs(row[j]) > abs(row[j-1]) and abs(row[j]) > abs(row[j+1]) and abs(row[j]) >= threshold): 
                t_list.append(time_arr[i])
                x_list.append(x_loc[j])
    return np.array(t_list), np.array(x_list)

def _best_next_point(t_s, x_s, last, c_min, c_max, max_dt_gap, dt_output):
    """Among points after `last`, return the index whose implied speed is
    closest to the middle of [c_min, c_max], or None if none qualify (use
    of Claude AI to construct this function)."""
    t_last, x_last  = t_s[last], x_s[last]
    best_j, best_dc = None, np.inf
    for j in range(last + 1, len(t_s)):
        dtt = t_s[j] - t_last
        if dtt > max_dt_gap*dt_output:
            # Because that value is not relevant
            break
        if dtt <= 0:
            continue
        c = (x_s[j] - x_last)/dtt
        if c_min <= c <= c_max:
            dc = abs(c - 0.5*(c_min + c_max))
            if dc < best_dc:
                best_j, best_dc = j, dc
    return best_j

def track_diagonal(peak_t, peak_x, c_min, c_max, dt_output, max_dt_gap=5):
    """Link peaks across timesteps into tracks moving at speed in [c_min, c_max]."""
    sorted_idx = np.argsort(peak_t)
    t_s = peak_t[sorted_idx]
    x_s = peak_x[sorted_idx]
    # Each candidate track is increased if a good value is obtained.
    all_tracks = []
    for i in range(len(t_s)):
        track_idx = [i]
        while True:
            best_j = _best_next_point(t_s, x_s, track_idx[-1], c_min, c_max,
                                       max_dt_gap, dt_output)
            if best_j is None:
                break
            track_idx.append(best_j)
        if len(track_idx) >= 2:
            all_tracks.append(tuple(track_idx))
    # Longest tracks are prioritised.
    # Short trackcs are kept only ig half the points have not been allocated
    # to a longer track.
    all_tracks.sort(key=len, reverse=True)
    used, tracks = set(), []
    for tr in all_tracks:
        fresh = [idx for idx in tr if idx not in used]
        if len(fresh) >= max(2, len(tr)//2):
            tracks.append([(t_s[k], x_s[k]) for k in tr])
            used.update(tr)
    return tracks

def fit_speeds(tracks):
    """Linear fit x(t) for each track, returning the slope (wave speed)."""
    speeds = []
    for tr in tracks:
        t_vals = np.array([p[0] for p in tr])
        x_vals = np.array([p[1] for p in tr])
        speeds.append(np.polyfit(t_vals, x_vals, 1)[0])
    return speeds

def filter_tracks(tracks, min_len, min_dx):
    """Keep only tracks with enough points and enough spatial extent to
    give a reliable speed fit."""
    return [tr for tr in tracks if len(tr) >= min_len and
            (max(p[1] for p in tr) - min(p[1] for p in tr)) >= min_dx]

def plot_tracks(ax, valid_dn, valid_up, x_min, x_max, time_array):
    for tr, color in [(t, "red") for t in valid_dn] + [(t, "blue") for t in valid_up]:
        t_vals = np.array([p[0] for p in tr])
        x_vals = np.array([p[1] for p in tr])
        coeffs = np.polyfit(t_vals, x_vals, 1)
        c = coeffs[0]
        # Clips the fitted line to the region's x-bounds and the data's
        # time range, so the segment doesn't extrapolate past the 
        # plotted axes
        t_lo = max(min((x_min - coeffs[1])/coeffs[0],
                        (x_max - coeffs[1])/coeffs[0]), time_array[0])
        t_hi = min(max((x_min - coeffs[1])/coeffs[0],
                        (x_max - coeffs[1])/coeffs[0]), time_array[-1])
        ax.plot(np.polyval(coeffs, [t_lo, t_hi]), [t_lo, t_hi],
                "-", color=color, lw=2.0, label=f"c={c:.3f}")

# Region-based runs -> Assitance of Claude AI
def analyze_region(name, x_min, x_max, thresh_factor, c_min_dn, c_max_dn, c_min_up, 
                   c_max_up, min_len, min_dx_dn, min_dx_up):
    """ Analyze each region by using an adaptive threshold which changes
        based on the region of focus.
    """
    x_mask = (x >= x_min) & (x <= x_max)
    x_local = x[x_mask]
    amp_local = amplitude_array[:, x_mask]
    # Each region has a different wave magnitude (multiple wave-structures)
    # so the threshold is adaptive.
    threshold = thresh_factor * np.nanmax(np.abs(amp_local))
    peak_t, peak_x = detect_peaks(amp_local, x_local, time_array, threshold)
    tracks_dn = track_diagonal(peak_t, peak_x, c_min_dn, c_max_dn, dt_output)
    tracks_up = track_diagonal(peak_t, peak_x, c_min_up, c_max_up, dt_output)
    valid_dn = filter_tracks(tracks_dn, min_len, min_dx_dn)
    valid_up = filter_tracks(tracks_up, min_len, min_dx_up)
    speeds_dn = fit_speeds(valid_dn)
    speeds_up = fit_speeds(valid_up)
    x_reg, t_reg = np.meshgrid(x_local, time_array)
    fig, ax = plt.subplots(figsize=(12, 6))
    cf = ax.contourf(x_reg, t_reg, amp_local, levels=100, cmap="RdBu_r")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Interface amplitude (m)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax.scatter(peak_x, peak_t, s=4, color="black", alpha=0.6,
               label=f"detected peaks (n={len(peak_t)})")
    plot_tracks(ax, valid_dn, valid_up, x_min, x_max, time_array)
    ax.set_xlabel("x (m)", fontsize=15); ax.set_ylabel("Time (s)", fontsize=15)
    ax.set_title(f"{name.replace('_', ' ').title()} — Wave Tracks", fontsize=17)
    ax.tick_params(labelsize=13)
    ax.set_xlim(x_local[0], x_local[-1]); ax.set_ylim(time_array[0], time_array[-1])
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc="upper right")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{name}_tracks.png"), dpi=300)
    plt.close(fig)
    return {
        "n_peaks": len(peak_t),
        "speeds_dn": speeds_dn,
        "n_tracks_dn": len(valid_dn), "mean_c_dn": np.mean(speeds_dn) if speeds_dn else np.nan,
        "std_c_dn": np.std(speeds_dn) if speeds_dn else np.nan,
        "speeds_up": speeds_up,
        "n_tracks_up": len(valid_up), "mean_c_up": np.mean(speeds_up) if speeds_up else np.nan,
        "std_c_up": np.std(speeds_up) if speeds_up else np.nan,
    }

def pct_err(measured, theoretical):
    """Percentage error of a measured speed against the theoretical value."""
    if np.isnan(measured) or theoretical == 0:
        return np.nan
    return 100.0*abs(measured - theoretical)/abs(theoretical)

results = {}
for region_args in REGIONS:
    name = region_args[0]
    r = analyze_region(*region_args)
    results[name] = r
    print(f"{name:12s}  peaks={r['n_peaks']:5d}"
          f"c+ tracks={r['n_tracks_dn']:3d} mean={r['mean_c_dn']:.4f} m/s"
          f"c- tracks={r['n_tracks_up']:3d} mean={r['mean_c_up']:.4f} m/s")
# Comparison with the linear dispersion relation
print()
print("="*78)
print("WAVE SPEED vs LINEAR THEORY")
print(f"  Theory:  c+ = {c_plus:.4f} m/s   c- = {c_minus:.4f} m/s")
print("="*78)
for name, r in results.items():
    err_dn = pct_err(r["mean_c_dn"], c_plus)
    err_up = pct_err(r["mean_c_up"], c_minus)
    print(f"  {name:12s}  c+ = {r['mean_c_dn']:.4f} m/s  (err {err_dn:.2f}%)"
          f"   c- = {r['mean_c_up']:.4f} m/s  (err {err_up:.2f}%)")
print("="*78)

# Summary plot: tracked wave speeds vs linear theory, per region (assistance of Claude AI)
region_labels = [args[0] for args in REGIONS]
x_pos = np.arange(len(region_labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
for ax, mean_key, std_key, speeds_key, c_theory, title in [
    (ax1, "mean_c_dn", "std_c_dn", "speeds_dn", c_plus,  "c+ (downstream-travelling)"),
    (ax2, "mean_c_up", "std_c_up", "speeds_up", c_minus, "c- (upstream-travelling)"),
]:
    band = abs(c_theory) * 0.05
    ax.axhspan(c_theory - band, c_theory + band, color="gray", alpha=0.15)
    ax.axhline(c_theory, color="black", lw=1.8, ls="--", label=f"Theory = {c_theory:.3f} m/s")
    for i, name in enumerate(region_labels):
        speeds = results[name][speeds_key]
        if speeds:
            ax.scatter([x_pos[i]]*len(speeds), speeds, s=20, color="steelblue", alpha=0.7)
    means = [results[name][mean_key] for name in region_labels]
    stds  = [results[name][std_key] for name in region_labels]
    valid = [i for i, m in enumerate(means) if not np.isnan(m)]
    ax.errorbar([x_pos[i] for i in valid], [means[i] for i in valid],
                yerr=[0.0 if np.isnan(stds[i]) else stds[i] for i in valid],
                fmt="o", color="steelblue", ms=9, lw=2.0, capsize=6,
                label=r"Tracker mean $\pm$ std")
    for i in valid:
        err = pct_err(means[i], c_theory)
        ax.annotate(f"{err:.1f}%", (x_pos[i], means[i]),
                    textcoords="offset points", xytext=(10, 6), fontsize=13)
    ax.set_xticks(x_pos); ax.set_xticklabels(region_labels, fontsize=14)
    ax.set_ylabel("Wave speed (m/s)", fontsize=15)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3, ls=":")
fig.suptitle("Tracked Wave Speeds vs Linear Theory", fontsize=18)
fig.tight_layout()
plt.savefig(os.path.join(OUTDIR, "wave_speed_summary.png"), dpi=300)
plt.close(fig)
