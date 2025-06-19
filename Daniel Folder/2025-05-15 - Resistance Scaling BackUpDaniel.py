# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:33:45 2025

@author: Daniël
"""

### WVA1 - Opdracht 2 (Handleiding no.7) - Sleeptankproef Practicum - Verwerking Meetdata Daniël ###

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ALL IN SI UNITS (unless specified differently!)

# CONSTANTS
GRAVACC = 9.81
RHO_SEAWATER_12C = 1026.6376 # kg/m**3
RHO_FRESHWATER_17p7C = 998.6536 # kg/m**3 # Density of Towing Tank Water at 17.7 Celsius
VISCOSITY_SEAWATER_12C = 1.28324*10**(-6) # m**2/s
VISCOSITY_FRESHWATER_17p7C = 1.0621*10**(-6) # m**2/s # Viscosity of Towing Tank Water at 17.7 Celsius

# Particulars of Labrax
Lwl = 117.7 # m
Lpp = 116.85 # m
B = 14.3 # m
Taft = 5.5 # m
Tfwd = 5.5 # m
S_wet_ship = 2661.0 # m**2
Deplacement = 7952.8 # m**3
C_b = 0.8591 # block coefficient
alpha = 50
v_ship_operational = 9.5*(1852/3600) # m/s
v_ship_max = 10.5*(1852/3600) # m/s
P_ship = 1200*10**3 #kW
R_ship_max = (0.7*P_ship)/v_ship_max # Assumption

# Meetdata Sleeptankproef van MT5 mei25 Labrax Model
run_no = [501, 502, 503, 504, 505, 506, 507, 509, 510, 511] # [-]
v_0 = (-0.0016)
R_0 = 0.8402
v_model_initial = [0.0973, 0.1968, 0.2957, 0.3948, 0.4941, 0.5934, 0.6925, 0.8904, 0.9896, 1.1880] # [m/s]
R_model_initial = [0.8984, 1.0203, 1.1544, 1.3701, 1.6332, 1.9424, 2.3537, 3.4843, 4.8205, 9.5683] # [N]
v_model, R_model = [], []
for v in range(0,len(v_model_initial)):
    v_model.append( v_model_initial[v]-v_0 )
    R_model.append( R_model_initial[v]-R_0 )
    
# Scaling
Lwl_model = Lwl/alpha
v_ship = []
for v in v_model:
    v_ship.append(v*math.sqrt(alpha))
S_wet_model = S_wet_ship/(alpha**2)
    
# Functions
def calcFroudeNumber(v_, L_):
    Fn_ = v_ / math.sqrt(GRAVACC*L_)
    return Fn_

def calcReynoldsNumber(v_, L_, viscosity_):
    Re_ = (v_*L_) / viscosity_
    return Re_
    
def calcCoeffFriction(Re_):
    C_f_ = 0.075 / ((math.log10(Re_)-2)**2)
    return C_f_

def calcCoeffWave(C_t_, C_f_, k_):
    C_w_ = C_t_ - (1+k_)*C_f_
    return C_w_

def calcCoeffTotal(R_, rho_, v_, S_wet_):
    C_t_ = R_ / (0.5*S_wet_*rho_*v_**2)
    return C_t_
    
# Calculation
Fn_model = []
Re_model = []
Cf_model = []
Ct_model = []
for i in range(0,len(run_no)):
    Fn_m = calcFroudeNumber(v_model[i], Lwl_model)
    Fn_model.append( Fn_m )
    Re_m = calcReynoldsNumber(v_model[i], Lwl_model, VISCOSITY_FRESHWATER_17p7C)
    Re_model.append( Re_m )
    Cf_m = calcCoeffFriction(Re_m)
    Cf_model.append( Cf_m )
    Ct_m = calcCoeffTotal(R_model[i], RHO_FRESHWATER_17p7C, v_model[i], S_wet_model)
    Ct_model.append( Ct_m )

Ctm_vs_Cfm = []
Fnm_vs_Cfm = []
for i in range(0,len(Ct_model)):
    if Fn_model[i] > 0.1 and Fn_model[i] < 0.2:
        Ctm_vs_Cfm.append( Ct_model[i]/Cf_model[i] )
        Fnm_vs_Cfm.append( (Fn_model[i]**4)/Cf_model[i] )
#plt.title("Prohaska Plot")
#plt.scatter(Fnm_vs_Cfm, Ctm_vs_Cfm)
#plt.xlabel("$Fn^4/C_f$")
#plt.ylabel("$C_t/C_f$")
#plt.grid(True)
#plt.show()
interpol_formfactor = interp1d(Fnm_vs_Cfm, Ctm_vs_Cfm, kind='linear', fill_value='extrapolate')
form_factor = interpol_formfactor(0.0) - 1 # Volgens methode dat de functie de ordinaat bij 1+k intersect.

Re_ship = []
Cf_ship = []
Ct_ship = []
Cw_ship = []
for i in range(0,len(run_no)):
    Re_p = calcReynoldsNumber(v_ship[i], Lwl, VISCOSITY_SEAWATER_12C)
    Re_ship.append( Re_p )
    Cf_p = calcCoeffFriction(Re_p)
    Cf_ship.append( Cf_p )
    Cw_ = calcCoeffWave(Ct_m, Cf_m, form_factor)
    Cw_ship.append(Cw_)
    Ct_p = Cw_ + (1+form_factor)*Cf_p
    Ct_ship.append( Ct_p )
    
# Resistance Calculation of Scaled Measurements from Experiment of MTgroup 5 from 2025 from Model to Prototype
R_total_ship = []
for i in range(0,len(run_no)):
    R_t_p = Ct_ship[i] * 0.5 * RHO_SEAWATER_12C * v_ship[i]**2 * S_wet_ship
    R_total_ship.append(R_t_p)

plt.title("Weerstandskromme van de Labrax")
plt.scatter(v_ship, np.array(R_total_ship)/1000, c='orange', marker='o')
plt.plot(v_ship, np.array(R_total_ship)/1000, label="Weerstandskromme uit sleeptankproef-meetdata")
plt.xlabel("geschaalde scheepssnelheid $v_s$ (in $[m/s]$)")
plt.ylabel("geschaalde scheepsweerstand $R_{total}$ (in $[kN]$)")
plt.grid(True)
plt.legend()
plt.show()

print("\n------- END RUN -------\n")
# BackUp Daniel #