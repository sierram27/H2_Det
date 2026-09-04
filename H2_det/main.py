# H2 Detonation #
import cantera as ct
import matplotlib.pyplot as plt
import sdtoolbox
import polars as pol

from sdtoolbox.postshock import CJspeed, PostShock_eq, PostShock_fr
from sdtoolbox.znd import zndsolve

# ------------------
# Initial Conditions
# ------------------

# Change script to allow for multiple scenarios to be evaluated
T = [0] * 10
for i in range(0,9):
    T[i] = 300.0 + 50.0 * i
    print(T[i])
    
P = [0] * 10
for i in range(0,9):
    if i==0:
        P[i] = ct.one_atm
    else:
        P[i] = 100000.0 + 50000.0*i
    print(P[i] )

# Static initial conditions
T1 = 300.0          # K
P1 = ct.one_atm     # Pa
phi = 0.5

mech = "ffcm2_h2.yaml" #sets mech to the mechanism gri30.yaml

# Lean H2-air at phi = 0.5
q = 'H2:1 O2:1 N2:3.76'

# ---------------------
# Check initial mixture
# ---------------------

gas1 = ct.Solution(mech); #sets gas1 to the solution mechanism
gas1.TPX = T1,P1,q

# -----------------------------------------
# Calculate Chapman-Jouget detonation speed
# -----------------------------------------

cj_speed = CJspeed(P1, T1, q, mech)

# --------------------------------------
# Calculate equilibrium CJ product state
# --------------------------------------

gas_cj = PostShock_fr(cj_speed, P1, T1, q, mech)

# -------------
# Print results
# -------------

print("\nCJ detonation")
print("-------------")
print(f"CJ speed = {cj_speed:.2f} m/s")
print(f"CJ pressure = {gas_cj.P/1e5:.3f} bar") #converts Pa to bar for display
print(f"CJ temperature = {gas_cj.T:.1f} K")
print(f"CJ density = {gas_cj.density:.4f} kg/m^3")

# ---------------------------------------------
# Frozen state immediately behind leading shock
# ---------------------------------------------

gas_shock = PostShock_fr(cj_speed, P1, T1, q, mech)

# ---------------
# Extract Results
# ---------------
znd = zndsolve(gas_shock, gas1, cj_speed, t_end=5.0e-5, advanced_output=True)

# ---------------
# Extract Results
# ---------------

x = znd["distance"]
T = znd["T"]
P = znd["P"]

h2_index = gas1.species_index("H2")
Y_H2 = znd["species"][h2_index, :]
#print(f"Y_H2 starting = " f"{Y_H2[1]}") - Returned 0.0144675 - good

# -----------------------
# Print Useful Quantities
# -----------------------

print("\nZND structure")
print("-------------")

if "ind_len_ZND" in znd:
    print(f"Induction length = " f"{znd['ind_len_ZND']:.6e} m")
if "ind_time_ZND" in znd:
    print(f"Induction time = " f"{znd['ind_time_ZND']:.6e} s")
if "exo_len_ZND" in znd:
    print(f"Exothermic length = " f"{znd['exo_len_ZND']:.6e} m")

# ----------------
# Plot Temperature
# ----------------

#plt.figure()
#plt.plot(x*1000.0, T)
#plt.xlabel("Distance behind shock [mm]")
#plt.ylabel("Temperature [K]")
#plt.tight_layout()
#plt.show()

# -------------
# Plot Pressure
# -------------

#plt.figure()
#plt.plot(x*1000.0, P)
#plt.xlabel("Distance behind shock [mm]")
#plt.ylabel("Pressure [bar]")
#plt.tight_layout()
#plt.show()

# ---------------------
# Plot H2 Mole Fraction
# ---------------------

#plt.figure()
#plt.plot(x*1000.0, Y_H2)
#plt.xlabel("Distance behind shock [mm]")
#plt.ylabel("H2 mole fraction")
#plt.tight_layout()
#plt.show()

