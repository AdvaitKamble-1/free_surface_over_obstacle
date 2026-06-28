# Code to plot quantitative Reichl data

# Author: Advait Kamble
# Date: 27/06/2026

import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "reichl_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File Format
#
# ns =        1      time =     0.0000010
#              Cd    4.178109440242330E+04
#              Cl   -6.684975104387719E-01
# VS Transverse P   -1.086607785016764E-03
# DA Streamwise P    5.061716864707195E-01
# DA Transverse P    1.154886609062087E-04

# TOGGLES to view plots, save plots, view print blocks

SHOW_PLOTS = {
    "DA_at_max_Cl"    : True,
    "fr_local_vs_hD"  : True,
    "St_vs_hD"        : True,
}

SAVE_PLOTS = {
    "DA_at_max_Cl"    : True,
    "fr_local_vs_hD"  : True,
    "St_vs_hD"        : True,
}

PRINT_BLOCKS = {
    "da_at_max_cl_table" : True,
    "fr_local_table"     : True,
    "st_table"           : True,
}

# PHYSICAL PARAMETERS
D = 1.0 # cylinder diameter
U = 1.0 # free-stream velocity
St_0 = 0.208356 # single-phase reference Strouhal number

# Mask
t_start_default = 5.0

# FILE GROUPS - keyed by global Froude number
groups = {
    0.25: {
        "h/D=0.25": {"file": "00001_reichl_025_025.txt", "load": True},
        "h/D=0.5":  {"file": "00001_reichl_05_025.txt",  "load": True},
        "h/D=1.0":  {"file": "00001_reichl_1_025.txt",   "load": True},
        "h/D=1.5":  {"file": "00001_reichl_15_025.txt",  "load": True},
        "h/D=2.0":  {"file": "00001_reichl_2_025.txt",   "load": True},
        "h/D=2.5":  {"file": "00001_reichl_25_025.txt",  "load": True},
    },
    0.30: {
        "h/D=0.25": {"file": "00001_reichl_025_03.txt",  "load": True},
        "h/D=0.5":  {"file": "00001_reichl_05_03.txt",   "load": True},
        "h/D=1.0":  {"file": "00001_reichl_1_03.txt",    "load": True},
        "h/D=1.5":  {"file": "00001_reichl_15_03.txt",   "load": True},
        "h/D=2.0":  {"file": "00001_reichl_2_03.txt",    "load": True},
        "h/D=2.5":  {"file": "00001_reichl_25_03.txt",   "load": True},
    },
    0.35: {
        "h/D=0.25": {"file": "00001_reichl_025_035.txt", "load": True},
        "h/D=0.5":  {"file": "00001_reichl_05_035.txt",  "load": True},
        "h/D=1.0":  {"file": "00001_reichl_1_035.txt",   "load": True},
        "h/D=1.5":  {"file": "00001_reichl_15_035.txt",  "load": True},
        "h/D=2.0":  {"file": "00001_reichl_2_035.txt",   "load": True},
        "h/D=2.5":  {"file": "00001_reichl_25_035.txt",  "load": True},
    },
    0.40: {
        "h/D=0.25": {"file": "00001_reichl_025_04.txt",  "load": True},
        "h/D=0.5":  {"file": "00001_reichl_05_04.txt",   "load": True},
        "h/D=1.0":  {"file": "00001_reichl_1_04.txt",    "load": True},
        "h/D=1.5":  {"file": "00001_reichl_15_04.txt",   "load": True},
        "h/D=2.0":  {"file": "00001_reichl_2_04.txt",    "load": True},
        "h/D=2.5":  {"file": "00001_reichl_25_04.txt",   "load": True},
    },
}

# PARSER -> written through assitance of Claude
def parse_file(filename):
    ns, times, Cd, Cl, VS_trans, DA_stream, DA_trans = [], [], [], [], [], [], []
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
            elif s.startswith("DA Streamwise P"):
                DA_stream.append(float(s.split("DA Streamwise P")[1].strip()))
            elif s.startswith("DA Transverse P"):
                DA_trans.append(float(s.split("DA Transverse P")[1].strip()))
    return {
        "ns": np.array(ns),
        "times": np.array(times),
        "Cd": np.array(Cd),
        "Cl": np.array(Cl),
        "VS_trans": np.array(VS_trans),
        "DA_stream": np.array(DA_stream),
        "DA_trans": np.array(DA_trans),
    }

# LOAD
data = {} # data[fr][label] = parsed dict
for fr, files in groups.items():
    data[fr] = {}
    for label, cfg in files.items():
        if cfg["load"]:
            data[fr][label] = parse_file(cfg["file"])

def t_start_for(fr, label):
    return groups[fr][label].get("t_start", t_start_default)

# Plotting for avg velocity at max lift vs h/D
if SHOW_PLOTS["DA_at_max_Cl"] or SAVE_PLOTS["DA_at_max_Cl"]:
    fig, ax = plt.subplots(figsize=(7, 5))
    for fr, fr_data in data.items():
        gap_ratios = []
        DA_stream_at_max_Cl = []
        for label, d in fr_data.items():
            ts = t_start_for(fr, label)
            mask = d["times"] >= ts
            Cl_masked = d["Cl"][mask]
            DA_stream_masked = d["DA_stream"][mask]
            idx_max_Cl = np.argmax(Cl_masked)
            hD = float(label.split("=")[1])
            gap_ratios.append(hD)
            DA_stream_at_max_Cl.append(DA_stream_masked[idx_max_Cl])
            print(f"Fr={fr:.2f}  {label}: max Cl = {Cl_masked[idx_max_Cl]:.4f}"
                  f"  |  DA streamwise vel at max Cl = {DA_stream_masked[idx_max_Cl]:.6f}")
        gap_ratios = np.array(gap_ratios)
        DA_stream_at_max_Cl = np.array(DA_stream_at_max_Cl)
        sort_idx = np.argsort(gap_ratios)
        gap_ratios = gap_ratios[sort_idx]
        DA_stream_at_max_Cl = DA_stream_at_max_Cl[sort_idx]
        ax.plot(gap_ratios, DA_stream_at_max_Cl, marker='o', linewidth=1.0, label=f"Fr = {fr:.2f}")
    ax.set_xlabel("Gap ratio $h/D$")
    ax.set_ylabel("DA Streamwise Velocity at peak $C_l$")
    ax.set_title("Average streamwise velocity above cylinder at max lift")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "reichl_DA_stream_at_max_Cl_vs_hD.png")
    if SAVE_PLOTS["DA_at_max_Cl"]:
        fig.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    if SHOW_PLOTS["DA_at_max_Cl"]:
        plt.show()
    plt.close(fig)

# Plotting for local Fr vs h/D
if SHOW_PLOTS["fr_local_vs_hD"] or SAVE_PLOTS["fr_local_vs_hD"]:
    fig, ax = plt.subplots(figsize=(7, 5))
    for fr, fr_data in data.items():
        gap_ratios_fr = []
        Fr_local_vals = []
        for label, d in fr_data.items():
            ts = t_start_for(fr, label)
            mask = d["times"] >= ts
            DA_stream_masked = d["DA_stream"][mask]
            hD = float(label.split("=")[1])
            U_max = np.max(DA_stream_masked)
            Fr_local = U_max * fr / np.sqrt(hD)
            gap_ratios_fr.append(hD)
            Fr_local_vals.append(Fr_local)
            print(f"Fr={fr:.2f}  {label}: max DA_stream = {U_max:.6f}  |  Fr_local = {Fr_local:.4f}")
        gap_ratios_fr = np.array(gap_ratios_fr)
        Fr_local_vals = np.array(Fr_local_vals)
        sort_idx = np.argsort(gap_ratios_fr)
        gap_ratios_fr = gap_ratios_fr[sort_idx]
        Fr_local_vals = Fr_local_vals[sort_idx]
        ax.plot(gap_ratios_fr, Fr_local_vals, marker='o', linewidth=1.0, label=f"Fr = {fr:.2f}")
    ax.set_xlabel("Gap ratio $h/D$")
    ax.set_ylabel("Local Froude number $Fr_{local}$")
    ax.set_title("Local Froude number above cylinder vs gap ratio")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "reichl_Fr_local_vs_hD.png")
    if SAVE_PLOTS["fr_local_vs_hD"]:
        fig.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    if SHOW_PLOTS["fr_local_vs_hD"]:
        plt.show()
    plt.close(fig)

# SUMMARY TABLES -> assistance of Claude
# Precompute both metrics for all Fr and h/D combinations
_fr_list = sorted(data.keys())
_hD_labels = sorted(
    {label for fr_data in data.values() for label in fr_data},
    key=lambda s: float(s.split("=")[1])
)

da_table = {fr: {} for fr in _fr_list}
fr_table = {fr: {} for fr in _fr_list}
st_table = {fr: {} for fr in _fr_list}
for fr, fr_data in data.items():
    for label, d in fr_data.items():
        ts = t_start_for(fr, label)
        mask = d["times"] >= ts
        Cl_masked = d["Cl"][mask]
        times_masked = d["times"][mask]
        DA_stream_masked = d["DA_stream"][mask]
        idx_max_Cl = np.argmax(Cl_masked)
        hD = float(label.split("=")[1])
        U_max = np.max(DA_stream_masked)
        da_table[fr][label] = DA_stream_masked[idx_max_Cl]
        fr_table[fr][label] = U_max * fr / np.sqrt(hD)
        if len(Cl_masked) >= 4:
            dt_cl = np.median(np.diff(times_masked))
            fft_mag = np.abs(np.fft.rfft(Cl_masked - np.mean(Cl_masked)))
            freqs = np.fft.rfftfreq(len(Cl_masked), d=dt_cl)
            fft_mag[0] = 0
            st_table[fr][label] = freqs[np.argmax(fft_mag)] * D / U
        else:
            st_table[fr][label] = float("nan")
col_w = 12
if PRINT_BLOCKS["da_at_max_cl_table"]:
    W = 8 + col_w * len(_fr_list)
    print("\n" + "="*W)
    print("DA STREAMWISE VELOCITY AT PEAK Cl")
    print("="*W)
    header = f"  {'h/D':>5}"
    for fr in _fr_list:
        header += f"  {'Fr='+f'{fr:.2f}':>{col_w-2}}"
    print(header)
    print("-"*W)
    for label in _hD_labels:
        hD  = float(label.split("=")[1])
        row = f"  {hD:>5.2f}"
        for fr in _fr_list:
            val = da_table[fr].get(label, float("nan"))
            row += f"  {val:>{col_w-2}.6f}"
        print(row)
    print("="*W)
if PRINT_BLOCKS["fr_local_table"]:
    W = 8 + col_w * len(_fr_list)
    print("\n" + "="*W)
    print("LOCAL FROUDE NUMBER  (Fr_local = U_max * Fr_global / sqrt(h/D))")
    print("="*W)
    header = f"  {'h/D':>5}"
    for fr in _fr_list:
        header += f"  {'Fr='+f'{fr:.2f}':>{col_w-2}}"
    print(header)
    print("-"*W)
    for label in _hD_labels:
        hD  = float(label.split("=")[1])
        row = f"  {hD:>5.2f}"
        for fr in _fr_list:
            val = fr_table[fr].get(label, float("nan"))
            row += f"  {val:>{col_w-2}.6f}"
        print(row)
    print("="*W)
if PRINT_BLOCKS["st_table"]:
    W = 8 + col_w * len(_fr_list)
    print("\n" + "="*W)
    print(f"STROUHAL NUMBER  (St = f_peak·D/U,  St₀ = {St_0})")
    print("="*W)
    header = f"  {'h/D':>5}"
    for fr in _fr_list:
        header += f"  {'Fr='+f'{fr:.2f}':>{col_w-2}}"
    print(header)
    print("-"*W)
    for label in _hD_labels:
        hD  = float(label.split("=")[1])
        row = f"  {hD:>5.2f}"
        for fr in _fr_list:
            val = st_table[fr].get(label, float("nan"))
            row += f"  {val:>{col_w-2}.6f}" if not np.isnan(val) else f"  {'N/A':>{col_w-2}}"
        print(row)
    print("-"*W)
    row = f"  {'St/St₀':>5}"
    for fr in _fr_list:
        row += f"  {'---':>{col_w-2}}"
    print(row)
    for label in _hD_labels:
        hD  = float(label.split("=")[1])
        row = f"  {hD:>5.2f}"
        for fr in _fr_list:
            val = st_table[fr].get(label, float("nan"))
            ratio = val / St_0
            row += f"  {ratio:>{col_w-2}.6f}" if not np.isnan(val) else f"  {'N/A':>{col_w-2}}"
        print(row)
    print("="*W)

# Plotting for St vs h/D
if SHOW_PLOTS["St_vs_hD"] or SAVE_PLOTS["St_vs_hD"]:
    fig, ax = plt.subplots(figsize=(7, 5))
    for fr in _fr_list:
        hD_vals  = []
        St_ratios = []
        for label in _hD_labels:
            val = st_table[fr].get(label, float("nan"))
            if not np.isnan(val):
                hD_vals.append(float(label.split("=")[1]))
                St_ratios.append(val / St_0)
        ax.plot(hD_vals, St_ratios, marker='o', linewidth=1.0, label=f"Fr = {fr:.2f}")
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label=f"St/St0 = 1  (St0 = {St_0})")
    ax.set_xlabel("Gap ratio $h/D$")
    ax.set_ylabel("$St / St_0$")
    ax.set_title("Strouhal number normalised by single-phase reference vs gap ratio")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "reichl_St_vs_hD.png")
    if SAVE_PLOTS["St_vs_hD"]:
        fig.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    if SHOW_PLOTS["St_vs_hD"]:
        plt.show()
    plt.close(fig)
