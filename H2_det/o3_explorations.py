# H2 detonations with O3 addition, sweep 0-9000 ppm O3:
#   1) lean H2-air (phi = 0.5)
#   2) stoichiometric H2-O2 diluted with 70% Ar
# Both mixtures are plotted on the same graphs.

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os

from sdtoolbox.postshock import CJspeed, PostShock_eq, PostShock_fr
from sdtoolbox.znd import zndsolve


# ------------------
# Initial Conditions
# ------------------

T0 = 300.0          # K
P0 = 101350.0       # Pa (101.35 kPa)

mech = "ffcm1_h2_o3_12sp_princeton.yaml"

# O3 cases [ppm]: 0, 1000, ..., 9000
o3_ppm = np.arange(0, 10000, 1000)

# Zone end time for the ZND integration [s]
t_end = 5.0e-5

# Mixtures without O3: label -> mole numbers
mixtures = {
    r"H$_2$-air, $\phi$ = 0.5":    {"H2": 1.0, "O2": 1.0, "N2": 3.76},
    r"2H$_2$ + O$_2$ + 7Ar":       {"H2": 1.0, "O2": 1.0, "AR": 7.0},
}

# Short names used in the CSV file names
short_names = ["h2_air_phi05", "h2_o2_ar"]


def mixture(base, ppm):
    """Add O3 to a base mixture.

    ppm is the O3 mole fraction in the oxidizer stream (everything except H2,
    plus the O3), the same convention as before. For H2-air at 1000 ppm this
    gives O3:0.00476476.
    """
    x = ppm * 1e-6
    n_ox = sum(v for s, v in base.items() if s != "H2")
    n_o3 = x * n_ox / (1.0 - x)
    return " ".join(f"{s}:{v}" for s, v in base.items()) + f" O3:{n_o3:.8f}"


# Make sure the mechanism contains every species we use
species = set(ct.Solution(mech).species_names)
needed = {s for base in mixtures.values() for s in base} | {"O3"}
missing = needed - species
if missing:
    raise ValueError(f"Mechanism {mech} is missing species: {sorted(missing)}")


# ---------------------
# Loop for calculations
# ---------------------

n = len(o3_ppm)
results = np.empty([len(mixtures), n, 7])
distances = []
species = []

for m, (label, base) in enumerate(mixtures.items()):
    print(f"\n=== {short_names[m]} ===")
    for k, ppm in enumerate(o3_ppm):
        q = mixture(base, ppm)
        print(f"Case {k + 1}/{n}: {ppm:5d} ppm O3  ->  {q}")

        # Initial mixture
        gas = ct.Solution(mech)
        gas.TPX = T0, P0, q

        # CJ detonation speed
        cj_speed = CJspeed(P0, T0, q, mech)

        # Equilibrium CJ product state
        gas_cj = PostShock_eq(cj_speed, P0, T0, q, mech)

        # Frozen state immediately behind the leading shock
        gas_shock = PostShock_fr(cj_speed, P0, T0, q, mech)

        # Integrate the ZND equations
        znd = zndsolve(gas_shock, gas, cj_speed, t_end=t_end, advanced_output=True)

        results[m, k, 0] = cj_speed                # CJ speed [m/s]
        results[m, k, 1] = gas_cj.P                # CJ pressure [Pa]
        results[m, k, 2] = gas_cj.T                # CJ temperature [K]
        results[m, k, 3] = gas_cj.density          # CJ density [kg/m^3]
        results[m, k, 4] = znd['ind_len_ZND']      # ZND induction length [m]
        results[m, k, 5] = znd['ind_time_ZND']     # ZND induction time [s]
        results[m, k, 6] = znd['exo_len_ZND']      # ZND exothermic length [m]
        distances.append(znd['distance'])       # distance behind shock front
        species.append(znd['species'])          # species mass fractions

        print(f"    D_CJ = {cj_speed:.1f} m/s, ind. length = {znd['ind_len_ZND'] * 1e3:.4f} mm")


# ------------------------------
# Save as csv (one per mixture)
# ------------------------------

header = ("O3_ppm,CJ_speed_m_per_s,CJ_pressure_Pa,CJ_temperature_K,CJ_density_kg_per_m3,"
          "ZND_induction_length_m,ZND_induction_time_s,ZND_exothermic_length_m")
for m, name in enumerate(short_names):
    np.savetxt(f"o3_sweep_results_{name}.csv", np.column_stack([o3_ppm, results[m]]),
               delimiter=",", header=header, comments="", fmt="%.6e")


# -----
# Plots
# -----

plt.rcParams.update({'font.size': 10})
markers = ["o-", "s--"]

# (column, scale factor, y-label, title, file name, is a length/time)
plots = [
    (4, 1e3,  "ZND Induction Length [mm]",    "ZND Induction Length vs. O$_3$ Addition",  "induction_length",  True),
    (5, 1e6,  "ZND Induction Time [$\\mu$s]", "ZND Induction Time vs. O$_3$ Addition",    "induction_time",    True),
    (6, 1e3,  "ZND Exothermic Length [mm]",   "ZND Exothermic Length vs. O$_3$ Addition", "exothermic_length", True),
    (0, 1.0,  "CJ Speed [m/s]",               "CJ Speed vs. O$_3$ Addition",              "cj_speed",          False),
    (2, 1.0,  "CJ Temperature [K]",           "CJ Temperature vs. O$_3$ Addition",        "cj_temperature",    False),
    (1, 1e-3, "CJ Pressure [kPa]",            "CJ Pressure vs. O$_3$ Addition",           "cj_pressure",       False),
    (3, 1.0,  "CJ Density [kg/m$^3$]",        "CJ Density vs. O$_3$ Addition",            "cj_density",        False),
]


try:
    _script = os.path.abspath(__file__)
    out_dir = os.path.join(os.path.dirname(_script),
                           os.path.splitext(os.path.basename(_script))[0] + "_figures")
except NameError:                     # e.g. running in a notebook
    out_dir = os.path.join(os.getcwd(), "o3_figures")
os.makedirs(out_dir, exist_ok=True)

LOG_Y_LENGTHS=True
for col, scale, ylabel, title, fname, is_length in plots:
    plt.figure()
    for m, label in enumerate(mixtures):
        plt.plot(o3_ppm, results[m, :, col] * scale, markers[m % len(markers)], label=label)
    if is_length and LOG_Y_LENGTHS:
        plt.yscale("log")
    plt.xlabel("O$_3$ in mixture [ppm]")
    plt.ylabel(ylabel)
    plt.title(f"{title}\n$T_0$ = {T0:.0f} K, $P_0$ = {P0 / 1e3:.2f} kPa")
    plt.xticks(o3_ppm)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{fname}_vs_O3")
    plt.savefig(fig_path + ".png", dpi=200)

#plt.show()

#Plot lengths on a not log scale to compare
LOG_Y_LENGTHS = False
for col, scale, ylabel, title, fname, is_length in plots:
    if is_length == True:
        plt.figure()
        for m, label in enumerate(mixtures):
            plt.plot(o3_ppm, results[m, :, col] * scale, markers[m % len(markers)], label=label)
        if is_length and LOG_Y_LENGTHS:
            plt.yscale("log")
        plt.xlabel("O$_3$ in mixture [ppm]")
        plt.ylabel(ylabel)
        plt.title(f"{title}\n$T_0$ = {T0:.0f} K, $P_0$ = {P0 / 1e3:.2f} kPa")
        plt.xticks(o3_ppm)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"{fname}_vs_O3_not_log")
        plt.savefig(fig_path + ".png", dpi=200)
#plt.show()

#plot all species on the same plot
species_to_plot = ['H2', 'O2', 'O3', 'H2O', 'OH', 'O', 'H', 'HO2', 'H2O2']
for k, ppm in enumerate(o3_ppm):
    plt.figure()
    for name in species_to_plot:
        index = gas.species_index(name)
        plt.plot(distances[k]*1e3, species[k][index,:], label=name)
    plt.xscale('log')
    plt.xlabel('Distance behind shock [mm]')
    plt.ylabel('Mass fraction')
    plt.title(f'Species mass fractions, {ppm} ppm O$_3$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'O3_all_species_{ppm}ppm.png', dpi=200)
#plt.show()