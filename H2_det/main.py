# H2 Detonation #
import cantera as ct
import matplotlib.pyplot as plt
import sdtoolbox
import polars as pl
import numpy as np

from sdtoolbox.postshock import CJspeed, PostShock_eq, PostShock_fr
from sdtoolbox.znd import zndsolve

# ------------------
# Initial Conditions
# ------------------

# Change script to allow for multiple scenarios to be evaluated
T_array = [0] * 10
for i in range(0,10):
    T_array[i] = 300.0 + 50.0 * i
    
P_array = [0] * 10
for i in range(0,10):
    if i==0:
        P_array[i] = ct.one_atm
    else:
        P_array[i] = 100000.0 + 50000.0*i

# Static initial conditions
T1 = 500.0          # K
P1 = 200000.0     # Pa
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

gas_cj = PostShock_eq(cj_speed, P1, T1, q, mech)

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

# -----------------------------------
# -----------------------------------
# USING ARRAYS FOR INITIAL CONDITIONS
# -----------------------------------
# -----------------------------------


# i belongs to temperature now
# j belongs to pressure now

results = np.empty([10,10,7])
    # index 0 = cj speed
    # 1 = cj pressure
    # 2 = cj temp
    # 3 = cj density
    # 4 = znd induction length
    # 5 = znd induction time
    # 6 = znd exothermic length

s_array_t = []
s_array_p = []

#Loop through temperature and pressure combinations
for j in range(0,10):
    #print(P_array[j])
    for i in range(0,10):
        
        # Check initial mixture
        gas2=ct.Solution(mech)
        gas2.TPX = T_array[i],P_array[j],q
        #print(T_array[i])
        
        # Calculate CJ detonation speed
        cj_speed2 = CJspeed(P_array[j], T_array[i], q, mech)
        
        # calculate equilibrium CJ product state
        gas_cj2 = PostShock_eq(cj_speed2, P_array[j], T_array[i], q, mech)
        
        # Frozen state immediately behind leading shock
        gas_shock2 = PostShock_fr(cj_speed2, P_array[j], T_array[i], q, mech)
        
        # Integrate znd equations
        znd2 = zndsolve(gas_shock2, gas2, cj_speed2, t_end=5.0e-5, advanced_output=True)
        
        # Extract znd results and store
        results[i,j,0] = cj_speed2              #CJ speed
        results[i,j,1] = gas_cj2.P              #CJ pressure
        results[i,j,2] = gas_cj2.T              #CJ temp
        results[i,j,3] = gas_cj2.density        #CJ density
        results[i,j,4] = znd2['ind_len_ZND']    #induction length
        results[i,j,5] = znd2['ind_time_ZND']   #induction time
        results[i,j,6] = znd2['exo_len_ZND']    #exothermic length

# Plot shit     
plt.figure()
for i,p in enumerate(P_array):
    plt.plot(
        T_array,
        results[:,i,0], 
        label=f"{p}"
    )
plt.xlabel("Temperature [K]")
plt.ylabel("CJ Speed [m/s]")
plt.legend()
plt.show()

plt.figure()
for i,t in enumerate(T_array):
    plt.plot(
        P_array,
        results[i,:,0], 
        label=f"{t}"
    )
plt.xlabel("Pressure (Pa)")
plt.ylabel("CJ Speed [m/s]")
plt.legend()
plt.show()
