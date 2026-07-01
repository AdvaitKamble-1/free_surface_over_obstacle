# Python Script to tabulate quantitative results from Bouscasse et al.

# Author: Advait Kamble
# Date: 27/06/2026

# AI Use:
# - Claude was used to assist in writing the parsing and filtering functions, as well as generating the summary tables and parity plots.
# - Claude was also used to help with formatting the output and ensuring that the code is clean and readable.

import numpy as np
import matplotlib.pyplot as plt
import os

# Domain Parameters
lb = 1

# Path to the results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bouscasse_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Start-time
t_start = 5.0

# Useful functions

def parse_file(filepath):
    """
    Parse the Bouscasse et al. data file and extract time, Cl, and Cd values.
    Returns:
        times: np.array of time values
        Cls: np.array of lift coefficient values
        Cds: np.array of drag coefficient values
    """
    times, Cls, Cds = [], [], []
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return np.array(times), np.array(Cls), np.array(Cds)
    # Read the file line by line
    with open(filepath, 'r') as f:
        current_time = None
        current_Cl= None
        current_Cd = None
        for line in f:
            # Strip whitespace and check for relevant lines
            stripped = line.strip()
            if 'ns =' in stripped and 'time =' in stripped:
                if current_time is not None and current_Cl is not None and current_Cd is not None:
                    times.append(current_time)
                    Cls.append(current_Cl)
                    Cds.append(current_Cd)
                parts = stripped.split('time =')
                current_time = float(parts[1].strip())
                current_Cl = None
                current_Cd = None
            elif stripped.startswith('Cl') or stripped.startswith('Lift Coeff'):
                parts = stripped.split()
                # AI: Handle cases where the line might not have enough parts
                if len(parts) >= 2:
                    try:
                        current_Cl = float(parts[-1])
                    except ValueError:
                        current_Cl = None
            elif stripped.startswith('Cd') or stripped.startswith('Drag Coeff'):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        current_Cd = float(parts[-1])
                    except ValueError:
                        current_Cd = None
    return np.array(times), np.array(Cls), np.array(Cds)


def filter_data(times, Cls, Cds, t_start, t_end=None, limit=10):
    """
    Filter the time series data based on the specified time window and limit for Cl and Cd values.
    Returns:
        t: np.array of filtered time values
        cl: np.array of filtered lift coefficient values
        cd: np.array of filtered drag coefficient values
    """
    mask = times >= t_start
    if t_end is not None:
        mask &= times <= t_end
    t, cl, cd = times[mask], Cls[mask], -Cds[mask]
    valid = ~(np.isnan(cl) | np.isnan(cd)) & (np.abs(cl) < limit) & (np.abs(cd) < limit)
    return t[valid], cl[valid], cd[valid]

# SEQUENCE 1 — h/D = 0.55, varying Fr
gap_seq1 = '055'
froude_numbers_1 = ['03', '06', '08', '1', '12', '16', '2']
froude_values_1 = [0.3, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]
t_windows_1 = {0.8: (5.0, 20.0), 1.0: (5.0, 26.0), 1.2: (5.0, 43.0)}
# Reference data from Bouscasse et al. for sequence 1
ref_data_1 = [(0.3, 1.61, -0.26, 0.60), (0.6, 1.44, -0.30, 0.03), (0.8, 1.26, -0.25, 0.03),
              (1.0, 1.13, -0.25, 0.01), (1.2, 1.19, -0.18, 0.03), (1.6, 1.03, -0.02, 0.03),
              (2.0, 0.92, -0.06, 0.07),
]
mean_Cls_seq1 = []
mean_Cds_seq1 = []
Clrms_seq1 = []
# Loop over the Froude numbers for sequence 1
for froude, fr_val in zip(froude_numbers_1, froude_values_1):
    filename = f'00001_bouscasse_{gap_seq1}_{froude}.txt'
    times, Cls, Cds = parse_file(filename)
    ts, te = t_windows_1.get(fr_val, (t_start, None))
    times_t, Cls_t, Cds_t = filter_data(times, Cls, Cds, ts, te)
    # Collect mean values and RMS for Cl and Cd
    if len(times_t) == 0:
        print(f"  WARNING: No valid data for Fr = {fr_val}, skipping.")
        mean_Cls_seq1.append(np.nan)
        mean_Cds_seq1.append(np.nan)
        Clrms_seq1.append(np.nan)
        continue
    mean_Cls_seq1.append(np.mean(Cls_t))
    mean_Cds_seq1.append(np.mean(Cds_t))
    Clrms_seq1.append(np.sqrt(np.mean(Cls_t**2)))
def _pct(fdm, ref):
    if np.isnan(ref):
        return f"{'–':>8}"
    return f"{100*(fdm - ref)/abs(ref):>7.2f}%"

def _fv(v):
    return f"{v:>9.4f}" if not np.isnan(v) else f"{'–':>9}"

# seq1_comparison
W = 108
print("\n" + "="*W)
print("SEQUENCE 1 - FDM vs Bouscasse et al., h/D = 0.55, varying Fr")
print("="*W)
print(f"  {'Frd':>5} "
      f"{'Cd_FDM':>9} {'Cd_Ref':>9} {'dCd%':>8} "
      f"{'Cl_FDM':>9} {'Cl_Ref':>9} {'dCl%':>8} "
      f"{'Clrms_FDM':>10} {'Clrms_Ref':>10} {'dClrms%':>8}")
print("-"*W)
for (frd, cd_r, cl_r, clrms_r), mean_Cd, mean_Cl, clrms in zip(
        ref_data_1, mean_Cds_seq1, mean_Cls_seq1, Clrms_seq1):
    print(f"  {frd:>5.2f} "
          f"{_fv(mean_Cd)} {_fv(cd_r)} {_pct(mean_Cd, cd_r)} "
          f"{_fv(mean_Cl)} {_fv(cl_r)} {_pct(mean_Cl, cl_r)} "
          f"{_fv(clrms):>10} {_fv(clrms_r):>10} {_pct(clrms, clrms_r):>8}")
print("="*W)
    
# SEQUENCE 2 — Fr = 1.0, varying h/D
gap_ratios_str_2 = ['m05', '0', '05', '1', '15', '25']
gap_ratios_val_2 = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.5]
froude_fixed_2 = '1'
t_windows_2 = {-0.5: (5.0, 30.0), 0.0: (5.0, 27.0), 0.5: (5.0, 27.0)}
ref_data_2 = [(-0.5, 0.45, -0.74, 0.06), (0.0, 1.26, -1.00, 0.06), (0.5, 1.52, -0.48, 0.06),
              (1.0, 1.46, -0.17, 0.12), (1.5, 1.54, -0.08, 0.38), (2.5, 1.58, -0.02, 0.57),
]
mean_Cls_seq2 = []
mean_Cds_seq2 = []
Clrms_seq2 = []
# Loop over the gap ratios for sequence 2
for gap, gap_val in zip(gap_ratios_str_2, gap_ratios_val_2):
    filename = f'00001_bouscasse_{gap}_{froude_fixed_2}.txt'
    times, Cls, Cds = parse_file(filename)
    ts, te = t_windows_2.get(gap_val, (t_start, None))
    times_t, Cls_t, Cds_t = filter_data(times, Cls, Cds, ts, te)
    # Collect mean values and RMS for Cl and Cd
    if len(times_t) == 0:
        print(f"  WARNING: No valid data for h/D = {gap_val}, skipping.")
        mean_Cls_seq2.append(np.nan)
        mean_Cds_seq2.append(np.nan)
        Clrms_seq2.append(np.nan)
        continue
    mean_Cls_seq2.append(np.mean(Cls_t))
    mean_Cds_seq2.append(np.mean(Cds_t))
    Clrms_seq2.append(np.sqrt(np.mean(Cls_t**2)))
# seq2_comparison
print("\n" + "="*W)
print("SEQUENCE 2 - FDM vs Bouscasse et al., Fr = 1.0, varying h/D")
print("="*W)
print(f"  {'h/D':>5} "
      f"{'Cd_FDM':>9} {'Cd_Ref':>9} {'dCd%':>8} "
      f"{'Cl_FDM':>9} {'Cl_Ref':>9} {'dCl%':>8} "
      f"{'Clrms_FDM':>10} {'Clrms_Ref':>10} {'dClrms%':>8}")
print("-"*W)
for (hd, cd_r, cl_r, clrms_r), mean_Cd, mean_Cl, clrms in zip(
        ref_data_2, mean_Cds_seq2, mean_Cls_seq2, Clrms_seq2):
    print(f"  {hd:>5.2f} "
          f"{_fv(mean_Cd)} {_fv(cd_r)} {_pct(mean_Cd, cd_r)} "
          f"{_fv(mean_Cl)} {_fv(cl_r)} {_pct(mean_Cl, cl_r)} "
          f"{_fv(clrms):>10} {_fv(clrms_r):>10} {_pct(clrms, clrms_r):>8}")
print("="*W)

# SEQUENCE 3 — h/D = -0.5, varying Fr
gap_fixed_3 = 'm05'
froude_numbers_3 = ['04', '06', '1', '16', '2']
froude_values_3 = [0.4, 0.6, 1.0, 1.6, 2.0]
t_windows_3 = {1.0: (5.0, 30.0)}
ref_data_3 = [(0.4, 0.35, -0.24, 0.06), (0.6, 0.25, -0.58, 0.09), (1.0, 0.40, -0.66, 0.08),
              (1.6, 0.48, -0.44, 0.09), (2.0, 0.54, -0.11, 0.03),
]
mean_Cls_seq3 = []
mean_Cds_seq3 = []
Clrms_seq3 = []
for froude, fr_val in zip(froude_numbers_3, froude_values_3):
    filename = f'00001_bouscasse_{gap_fixed_3}_{froude}.txt'
    times, Cls, Cds = parse_file(filename)
    ts, te = t_windows_3.get(fr_val, (t_start, None))
    times_t, Cls_t, Cds_t = filter_data(times, Cls, Cds, ts, te)
    if len(times_t) == 0:
        print(f"  WARNING: No valid data for Fr = {fr_val}, skipping.")
        mean_Cls_seq3.append(np.nan)
        mean_Cds_seq3.append(np.nan)
        Clrms_seq3.append(np.nan)
        continue
    mean_Cls_seq3.append(np.mean(Cls_t))
    mean_Cds_seq3.append(np.mean(Cds_t))
    Clrms_seq3.append(np.sqrt(np.mean(Cls_t**2)))
# seq3_comparison
print("\n" + "="*W)
print("SEQUENCE 3 - FDM vs Bouscasse et al., h/D = -0.5, varying Fr")
print("="*W)
print(f"  {'Frd':>5} "
      f"{'Cd_FDM':>9} {'Cd_Ref':>9} {'dCd%':>8} "
      f"{'Cl_FDM':>9} {'Cl_Ref':>9} {'dCl%':>8} "
      f"{'Clrms_FDM':>10} {'Clrms_Ref':>10} {'dClrms%':>8}")
print("-"*W)
for (frd, cd_r, cl_r, clrms_r), mean_Cd, mean_Cl, clrms in zip(
        ref_data_3, mean_Cds_seq3, mean_Cls_seq3, Clrms_seq3):
    print(f"  {frd:>5.2f} "
          f"{_fv(mean_Cd)} {_fv(cd_r)} {_pct(mean_Cd, cd_r)} "
          f"{_fv(mean_Cl)} {_fv(cl_r)} {_pct(mean_Cl, cl_r)} "
          f"{_fv(clrms):>10} {_fv(clrms_r):>10} {_pct(clrms, clrms_r):>8}")
print("="*W)

# SEQUENCE 4 — Fr = 2.0, varying h/D
gap_list_4 = ['m05', '0', '06', '1', '15', '25']
gap_values_4 = [-0.5, 0.0, 0.6, 1.0, 1.5, 2.5]
fr_fixed_4 = '2'
ref_data_4 = [(-0.5, 0.54, -0.13, 0.03), (0.0, 0.69, -0.35, 0.03), (0.6, 0.92, -0.06, 0.07), 
              (1.0, 0.89, -0.09, 0.11), (1.5, 0.98, -0.04, 0.17), (2.5, 1.05, -0.01, 0.18),
]
mean_Cls_seq4 = []
mean_Cds_seq4 = []
Clrms_seq4 = []
for gap, gap_val in zip(gap_list_4, gap_values_4):
    filename = f'00001_bouscasse_{gap}_{fr_fixed_4}.txt'
    times, Cls, Cds = parse_file(filename)
    times_t, Cls_t, Cds_t = filter_data(times, Cls, Cds, t_start)
    if len(times_t) == 0:
        print(f"  WARNING: No valid data for h/D = {gap_val}, skipping.")
        mean_Cls_seq4.append(np.nan)
        mean_Cds_seq4.append(np.nan)
        Clrms_seq4.append(np.nan)
        continue
    mean_Cls_seq4.append(np.mean(Cls_t))
    mean_Cds_seq4.append(np.mean(Cds_t))
    Clrms_seq4.append(np.sqrt(np.mean(Cls_t**2)))
# seq4_comparison
print("\n" + "="*W)
print("SEQUENCE 4 - FDM vs Bouscasse et al., Fr = 2.0, varying h/D")
print("="*W)
print(f"  {'h/D':>5} "
      f"{'Cd_FDM':>9} {'Cd_Ref':>9} {'dCd%':>8} "
      f"{'Cl_FDM':>9} {'Cl_Ref':>9} {'dCl%':>8} "
      f"{'Clrms_FDM':>10} {'Clrms_Ref':>10} {'dClrms%':>8}")
print("-"*W)
for (hd, cd_r, cl_r, clrms_r), mean_Cd, mean_Cl, clrms in zip(
        ref_data_4, mean_Cds_seq4, mean_Cls_seq4, Clrms_seq4):
    print(f"  {hd:>5.2f} "
          f"{_fv(mean_Cd)} {_fv(cd_r)} {_pct(mean_Cd, cd_r)} "
          f"{_fv(mean_Cl)} {_fv(cl_r)} {_pct(mean_Cl, cl_r)} "
          f"{_fv(clrms):>10} {_fv(clrms_r):>10} {_pct(clrms, clrms_r):>8}")
print("="*W)

# Parity plots for all sequences
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Plotting the parity for C_d, C_l, and C_{l,rms} for all sequences
axes[0].scatter([r[1] for r in ref_data_1], mean_Cds_seq1, label="Seq1 h/D=0.55, vary Fr", color="tab:blue",   marker="o", zorder=3, s=55)
axes[0].scatter([r[1] for r in ref_data_2], mean_Cds_seq2, label="Seq2 Fr=1.0, vary h/D",  color="tab:orange", marker="s", zorder=3, s=55)
axes[0].scatter([r[1] for r in ref_data_3], mean_Cds_seq3, label="Seq3 h/D=-0.5, vary Fr", color="tab:green",  marker="^", zorder=3, s=55)
axes[0].scatter([r[1] for r in ref_data_4], mean_Cds_seq4, label="Seq4 Fr=2.0, vary h/D",  color="tab:red",    marker="D", zorder=3, s=55)
axes[1].scatter([r[2] for r in ref_data_1], mean_Cls_seq1, label="Seq1 h/D=0.55, vary Fr", color="tab:blue",   marker="o", zorder=3, s=55)
axes[1].scatter([r[2] for r in ref_data_2], mean_Cls_seq2, label="Seq2 Fr=1.0, vary h/D",  color="tab:orange", marker="s", zorder=3, s=55)
axes[1].scatter([r[2] for r in ref_data_3], mean_Cls_seq3, label="Seq3 h/D=-0.5, vary Fr", color="tab:green",  marker="^", zorder=3, s=55)
axes[1].scatter([r[2] for r in ref_data_4], mean_Cls_seq4, label="Seq4 Fr=2.0, vary h/D",  color="tab:red",    marker="D", zorder=3, s=55)
axes[2].scatter([r[3] for r in ref_data_1], Clrms_seq1, label="Seq1 h/D=0.55, vary Fr", color="tab:blue",   marker="o", zorder=3, s=55)
axes[2].scatter([r[3] for r in ref_data_2], Clrms_seq2, label="Seq2 Fr=1.0, vary h/D",  color="tab:orange", marker="s", zorder=3, s=55)
axes[2].scatter([r[3] for r in ref_data_3], Clrms_seq3, label="Seq3 h/D=-0.5, vary Fr", color="tab:green",  marker="^", zorder=3, s=55)
axes[2].scatter([r[3] for r in ref_data_4], Clrms_seq4, label="Seq4 Fr=2.0, vary h/D",  color="tab:red",    marker="D", zorder=3, s=55)
# Set up parity lines and labels for each subplot - AI Handle
for ax, qty in zip(axes, [r"$C_d$", r"$C_l$", r"$C_{l,rms}$"]):
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]) * 1.05
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect match")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Reference {qty}", fontsize=14)
    ax.set_ylabel(f"FDM {qty}",       fontsize=14)
    ax.set_title(f"Parity - {qty}",   fontsize=14)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", "box")
fig.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "summary_parity.png"), dpi=150)
plt.show()

# Short description on how the parity plots were generated::
# 1. The figure is set up with three subplots for C_d, C_l, and C_{l,rms}.
# 2. For each sequence, the reference values and FDM computed values are plotted as scatter points with different colors and markers.
# 3. A parity line (y=x) is added to each subplot to indicate perfect agreement between the reference and computed values.
# 4. Axes are labeled, and legends are added for clarity. The aspect ratio is set to equal for accurate comparison.
# 5. The figure is saved as a PNG file in the results directory and displayed.