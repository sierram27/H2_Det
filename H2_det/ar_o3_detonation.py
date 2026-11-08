# H2-argon (phi = 0.5) detonation with O3 addition: sweep 0-9000 ppm O3 #

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


def mixture(ppm):
    """Lean H2-air (phi = 0.5) with O3 added to the oxidizer.

    Same convention as the original script: ppm is the O3 mole fraction in the
    oxidizer (air + O3), H2 not included. For 1000 ppm this gives
    O3:0.00476476, the value used before.
    """
    x = ppm * 1e-6
    n_o3 = x * 8.0 / (1.0 - x)
    return f"H2:2 O2:1 AR:7 O3:{n_o3:.8f}"


# ---------------------
# Loop for calculations
# ---------------------

n = len(o3_ppm)
results = np.empty([n, 7])
distances = []
species = []

for k, ppm in enumerate(o3_ppm):
    q = mixture(ppm)
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

    results[k, 0] = cj_speed                # CJ speed [m/s]
    results[k, 1] = gas_cj.P                # CJ pressure [Pa]
    results[k, 2] = gas_cj.T                # CJ temperature [K]
    results[k, 3] = gas_cj.density          # CJ density [kg/m^3]
    results[k, 4] = znd['ind_len_ZND']      # ZND induction length [m]
    results[k, 5] = znd['ind_time_ZND']     # ZND induction time [s]
    results[k, 6] = znd['exo_len_ZND']      # ZND exothermic length [m]
    distances.append(znd['distance'])       # distance behind shock front
    species.append(znd['species'])          # species mass fractions

    print(f"    D_CJ = {cj_speed:.1f} m/s, ind. length = {znd['ind_len_ZND'] * 1e3:.4f} mm")


# -----------
# Save as csv
# -----------

header = ("O3_ppm,CJ_speed_m_per_s,CJ_pressure_Pa,CJ_temperature_K,CJ_density_kg_per_m3,"
          "ZND_induction_length_m,ZND_induction_time_s,ZND_exothermic_length_m,distance_m,species_mass_fraction")
np.savetxt("o3_sweep_results.csv", np.column_stack([o3_ppm, results]),
           delimiter=",", header=header, comments="", fmt="%.6e")
print(f'distance shape: {znd['distance'].shape}')
print(f'species shape: {znd['species'].shape}')

# -----
# Plots
# -----

plt.rcParams.update({'font.size': 10})

# (column, scale factor, y-label, title, file name)
plots = [
    (4, 1e3,  "ZND Induction Length [mm]",    "ZND Induction Length vs. O$_3$ Addition",  "induction_length"),
    (5, 1e6,  "ZND Induction Time [$\\mu$s]", "ZND Induction Time vs. O$_3$ Addition",    "induction_time"),
    (6, 1e3,  "ZND Exothermic Length [mm]",   "ZND Exothermic Length vs. O$_3$ Addition", "exothermic_length"),
    (0, 1.0,  "CJ Speed [m/s]",               "CJ Speed vs. O$_3$ Addition",              "cj_speed"),
    (2, 1.0,  "CJ Temperature [K]",           "CJ Temperature vs. O$_3$ Addition",        "cj_temperature"),
    (1, 1e-3, "CJ Pressure [kPa]",            "CJ Pressure vs. O$_3$ Addition",           "cj_pressure"),
    (3, 1.0,  "CJ Density [kg/m$^3$]",        "CJ Density vs. O$_3$ Addition",            "cj_density"),
]

try:
    _script = os.path.abspath(__file__)
    out_dir = os.path.join(os.path.dirname(_script),
                           os.path.splitext(os.path.basename(_script))[0] + "_figures")
except NameError:                     # e.g. running in a notebook
    out_dir = os.path.join(os.getcwd(), "o3_figures")
os.makedirs(out_dir, exist_ok=True)

for col, scale, ylabel, title, fname in plots:
    plt.figure()
    plt.plot(o3_ppm, results[:, col] * scale, "o-", color='orange')
    plt.xlabel("O$_3$ in mixture [ppm]")
    plt.ylabel(ylabel)
    plt.title(f"{title}\n$T_0$ = {T0:.0f} K, $P_0$ = {P0 / 1e3:.2f} kPa")
    plt.xticks(o3_ppm)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{fname}_vs_O3")
    plt.savefig(fig_path + ".png", dpi=200)

print(f"\nFigures and csv saved in {out_dir}")
#plt.show()

# plot species mass fractions

i_OH = gas.species_index('OH')
i_O3 = gas.species_index('O3')
i_O = gas.species_index('O')
i_H2 = gas.species_index('H2')
i_H = gas.species_index('H')
i_HO2 = gas.species_index('HO2')
i_H2O2 = gas.species_index('H2O2')
i_AR = gas.species_index('AR')
i_H2O = gas.species_index('H2O')

species_plots = [
    (i_OH, 1, "Mass Fraction OH", "Mass Fraction OH vs. Distance Behind Shock", "OH_mass_fraction"),
    (i_O3, 1, "Mass Fraction O3", "Mass Fraction O3 vs. Distance Behind Shock", "O3_mass_fraction"),
    (i_O, 1, "Mass Fraction O", "Mass Fraction O vs. Distance Behind Shock", "O_mass_fraction"),
    (i_H2, 1, "Mass Fraction H2", "Mass Fraction H2 vs. Distance Behind Shock", "H2_mass_fraction"),
    (i_H, 1, "Mass Fraction H", "Mass Fraction H vs. Distance Behind Shock", "H_mass_fraction"),
    (i_HO2, 1, "Mass Fraction HO2", "Mass Fraction HO2 vs. Distance Behind Shock", "HO2_mass_fraction"),
    (i_H2O2, 1, "Mass Fraction H2O2", "Mass Fraction H2O2 vs. Distance Behind Shock", "H2O2_mass_fraction"),
    (i_AR, 1, "Mass Fraction Ar", "Mass Fraction Ar vs. Distance Behind Shock", "Ar_mass_fraction"),
    (i_H2O, 1, "Mass Fraction H2O", "Mass Fraction H2O vs. Distance Behind Shock", "H2O_mass_fraction"),
]

for row, scale, ylabel, title, fname in species_plots:
    plt.figure()
    for k, ppm in enumerate(o3_ppm):
        plt.plot(distances[k] * 1e3, species[k][row,:]*scale, label=f'{ppm} ppm O$_3$')
    plt.xlabel('Distance behind shock [mm]')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
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
    plt.savefig(f'AR_O3_all_species_{ppm}ppm.png', dpi=200)
plt.show()