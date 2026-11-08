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

n = 10
T_array = np.empty([n])
for i in range(0,n):
    T_array[i] = 300.0 + 50.0 * i
    
P_array = np.empty([n])
for i in range(0,n):
    if i==0:
        P_array[i] = ct.one_atm
    else:
        P_array[i] = 100000.0 + 50000.0*i

mech = "ffcm2_h2.yaml" #sets mech to the mechanism gri30.yaml

# Lean H2-air at phi = 0.5
q = 'H2:1 O2:1 N2:3.76'

# ---------------------
# ---------------------
# Loop for calculations
# ---------------------
# ---------------------

# i belongs to temperature now
# j belongs to pressure now

results = np.empty([n,n,7])

#Loop through temperature and pressure combinations
for j in range(0,n):
    #print(P_array[j])
    for i in range(0,n):
        
        # Check initial mixture
        gas=ct.Solution(mech)
        gas.TPX = T_array[i],P_array[j],q
        
        # Calculate CJ detonation speed
        cj_speed = CJspeed(P_array[j], T_array[i], q, mech)
        
        # calculate equilibrium CJ product state
        gas_cj = PostShock_eq(cj_speed, P_array[j], T_array[i], q, mech)
        
        # Frozen state immediately behind leading shock
        gas_shock = PostShock_fr(cj_speed, P_array[j], T_array[i], q, mech)
        
        # Integrate znd equations
        znd = zndsolve(gas_shock, gas, cj_speed, t_end=5.0e-5, advanced_output=True)
        
        # Extract znd results and store
        results[i,j,0] = cj_speed               #CJ speed
        results[i,j,1] = gas_cj.P               #CJ pressure
        results[i,j,2] = gas_cj.T               #CJ temperature
        results[i,j,3] = gas_cj.density         #CJ density
        results[i,j,4] = znd['ind_len_ZND']     #znd induction length
        results[i,j,5] = znd['ind_time_ZND']    #znd induction time
        results[i,j,6] = znd['exo_len_ZND']     #znd exothermic length

# -----------
# Save as csv
# -----------

#np.savetxt("cj_speed_data.csv", results[:,:,0],fmt='%.2e', delimiter=",")
#np.savetxt("cj_pressure_data.csv", results[:,:,1], delimiter=",")
#np.savetxt("cj_temperature_data.csv", results[:,:,2],fmt='%.1e', delimiter=",")
#np.savetxt("cj_density_data.csv", results[:,:,3],fmt='%.4e', delimiter=",")
#np.savetxt("znd_induction_length_data.csv", results[:,:,4],fmt='%.6e', delimiter=",")
#np.savetxt("znd_induction_time_data.csv", results[:,:,5],fmt='%.6e', delimiter=",")
#np.savetxt("znd_exothermic_length_data.csv", results[:,:,6],fmt='%.6e', delimiter=",")

# ---------
# Plot shit
# ---------

plt.rcParams.update({'font.size':10})

# Plot temperature changes for CJ speed
plt.figure()
for i,p in enumerate(P_array):
    plt.plot(
        T_array,
        results[:,i,0], 
        label=f"{p}"
    )
plt.xlabel("Temperature [K]")
plt.ylabel("CJ Speed [m/s]")
plt.title("CJ Speed vs. Initial Temperature")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Plot pressure changes for CJ speed
plt.figure()
for i,t in enumerate(T_array):
    plt.plot(
        P_array,
        results[i,:,0], 
        label=f"{t}"
    )
plt.xlabel("Pressure (Pa)")
plt.ylabel("CJ Speed [m/s]")
plt.title("CJ Speed vs. Initial Pressure")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# Plot temperature changes for CJ pressure
plt.figure()
for i,p in enumerate(P_array):
    plt.plot(
        T_array,
        results[:,i,1], 
        label=f"{p}"
    )
plt.xlabel("Temperature [K]")
plt.ylabel("CJ Pressure [Pa]")
plt.title("CJ Pressure vs. Initial Temperature")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Plot pressure changes for CJ pressure
plt.figure()
for i,t in enumerate(T_array):
    plt.plot(
        P_array,
        results[i,:,1], 
        label=f"{t}"
    )
plt.xlabel("Pressure (Pa)")
plt.ylabel("CJ Pressure [Pa]")
plt.title("CJ Pressure vs. Initial Pressure")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# Plot temperature changes for CJ temperature
plt.figure()
for i,p in enumerate(P_array):
    plt.plot(
        T_array,
        results[:,i,2], 
        label=f"{p}"
    )
plt.xlabel("Temperature [K]")
plt.ylabel("CJ Temperature [K]")
plt.title("CJ Temperature vs. Initial Temperature")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Plot pressure changes for CJ temperature
plt.figure()
for i,t in enumerate(T_array):
    plt.plot(
        P_array,
        results[i,:,2], 
        label=f"{t}"
    )
plt.xlabel("Pressure (Pa)")
plt.ylabel("CJ Temperature [K]")
plt.title("CJ Temperature vs. Initial Pressure")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# Plot temperature changes for CJ density
plt.figure()
for i,p in enumerate(P_array):
    plt.plot(
        T_array,
        results[:,i,3], 
        label=f"{p}"
    )
plt.xlabel("Temperature [K]")
plt.ylabel("CJ Density [kg/m^3]")
plt.title("CJ Density vs. Initial Temperature")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Plot pressure changes for CJ density
plt.figure()
for i,t in enumerate(T_array):
    plt.plot(
        P_array,
        results[i,:,3], 
        label=f"{t}"
    )
plt.xlabel("Pressure (Pa)")
plt.ylabel("CJ Density [kg/m^3]")
plt.title("CJ Density vs. Initial Pressure")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# Plot temperature changes for ZND induction length
plt.figure()
for i,p in enumerate(P_array):
    plt.plot(
        T_array,
        results[:,i,4], 
        label=f"{p}"
    )
plt.xlabel("Temperature [K]")
plt.ylabel("ZND Induction Length [m]")
plt.title("ZND Induction Length vs. Initial Temperature")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Plot pressure changes for ZND induction length
plt.figure()
for i,t in enumerate(T_array):
    plt.plot(
        P_array,
        results[i,:,4], 
        label=f"{t}"
    )
plt.xlabel("Pressure (Pa)")
plt.ylabel("ZND Induction Length [m]")
plt.title("ZND Induction Length vs. Initial Pressure")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# Plot temperature changes for ZND induction time
plt.figure()
for i,p in enumerate(P_array):
    plt.plot(
        T_array,
        results[:,i,5], 
        label=f"{p}"
    )
plt.xlabel("Temperature [K]")
plt.ylabel("ZND Induction Time [s]")
plt.title("ZND Induction Time vs. Initial Temperature")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Plot pressure changes for ZND induction time
plt.figure()
for i,t in enumerate(T_array):
    plt.plot(
        P_array,
        results[i,:,5], 
        label=f"{t}"
    )
plt.xlabel("Pressure (Pa)")
plt.ylabel("ZND Induction Time [s]")
plt.title("ZND Induction Time vs. Initial Pressure")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# Plot temperature changes for ZND exothermic length
plt.figure()
for i,p in enumerate(P_array):
    plt.plot(
        T_array,
        results[:,i,6]*1000, 
        label=f"{p}"
    )
plt.xlabel("Temperature [K]")
plt.ylabel("ZND Exothermic Length [m]")
plt.title("ZND Exothermic Length vs. Initial Temperature")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Plot pressure changes for ZND exothermic length
plt.figure()
for i,t in enumerate(T_array):
    plt.plot(
        P_array,
        results[i,:,6]*1000, 
        label=f"{t}"
    )
plt.xlabel("Pressure (Pa)")
plt.ylabel("ZND Exothermic Length [m]")
plt.title("ZND Exothermic Length vs. Initial Pressure")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()