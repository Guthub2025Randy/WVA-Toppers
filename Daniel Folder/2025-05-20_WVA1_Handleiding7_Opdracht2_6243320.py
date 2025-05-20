# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:33:45 2025

@author: Daniël
"""

### WVA1 - Opdracht 2 (Handleiding no.7) - Sleeptankproef Practicum - Verwerking Meetdata ###

import math
import numpy as np
import matplotlib.pyplot as plt

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


# Meetdata Sleeptankproef Labrax Model
run_no = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511] # [-]
v_model = [0.0016, 0.0973, 0.1968, 0.2957, 0.3948, 0.4941, 0.5934, 0.6925, 0.7916, 0.8904, 0.9896, 1.1880] # [m/s]
R_total_model = [0.8402, 0.8984, 1.0203, 1.1544, 1.3701, 1.6332, 1.9424, 2.3537, 2.1885, 3.4843, 4.8205, 9.5683] # [N]


# Functions
def calcFroudeNumber(v_, L_):
    Fn_ = v_ / math.sqrt(GRAVACC*L_)
    return Fn_

def calcReynoldsNumber(v_, L_, viscosity_):
    Re_ = (v_*L_) / viscosity_
    return Re_
    
def calcCoeffFriction(Re_):
    C_f_ = 0.075 / (math.log10(Re_)-2)**2
    return C_f_

def calcCoeffTotal(R_, rho_, v_, S_wet_):
    C_t_ = R_ / (0.5*S_wet_*rho_*v_**2)
    return C_t_

def calcFormFactor(C_b_, L_, B_, T_):
    k_ = -0.095 + 25.6*( C_b_/((L_/B_)**2 * math.sqrt(B_/T_)) )
    return k_

def calcCoeffWave(C_t_, C_f_, k_):
    C_w_ = C_t_ - (1+k_)*C_f_
    return C_w_

def scaleResistanceModelToShip(C_w_model_, C_f_model_, k_model, rho_, v_ship_, alpha_, S_wet_ship):
    R_ship_ = (C_w_model_ + C_f_model_*(1+k_model)) * 0.5*rho_*(v_ship_**2)*(S_wet_ship)
    return R_ship_


# Scaling
v_ship = []
for v in v_model:
    v_ship.append(v*math.sqrt(alpha))
Lwl_model = Lwl/alpha
B_model = B/alpha
Taft_model = Taft/alpha
Tfwd_model = Tfwd/alpha
S_wet_model = S_wet_ship/(alpha**2)
form_factor = calcFormFactor(C_b, Lwl, B, (Taft+Tfwd)/2)
form_factor_model = calcFormFactor(C_b, Lwl_model, B_model, (Taft_model+Tfwd_model)/2) # Block coefficient of model remains equal with that of prototype
print("\nScaled Velocity from Model to Ship (in [m/s]) = \n", v_ship)
print("\nScaled Length between Waterline from Ship to Model (in [m]) = \n", Lwl_model)
print("\nScaled Wet Surface Area from Ship to Model (in [m^2]) = \n", S_wet_model)
print("\nForm Factor k of Ship == Scaled Form Factor k of Model = \n", form_factor)


# Model Data
Fn_model = []
Re_model = []
Cf_model = []
Ct_model = []
for i, run in enumerate(run_no):
    Fn_m = calcFroudeNumber(v_model[i], Lwl_model)
    Fn_model.append( Fn_m )
    Re_m = calcReynoldsNumber(v_model[i], Lwl_model, VISCOSITY_FRESHWATER_17p7C)
    Re_model.append( Re_m )
    Cf_m = calcCoeffFriction(Re_m)
    Cf_model.append( Cf_m )
    Ct_m = calcCoeffTotal(R_total_model[i], RHO_FRESHWATER_17p7C, v_model[i], S_wet_model)
    Ct_model.append( Ct_m )

print("\nThe Froude Number of the Model = \n", Fn_model)
print("\nThe Reynold Number of the Model = \n", Re_model)
print("\nThe Coefficient of Frictional Resistance of the Model = \n", Cf_model)
print("\nThe Coefficient of Total Resistance of the Model = \n", Ct_model)
    
    
# Common Data (Model & Prototype)
Cw = []
for i, run in enumerate(run_no):
    Cw_both = calcCoeffWave(Ct_m, Cf_m, form_factor)
    Cw.append( Cw_both )
    
print("\nThe Coefficient of Wave Resistance of both (Model & Prototype) = \n", Fn_model)

    
# Ship Data
Fn_ship = []
Re_ship = []
Cf_ship = []
Ct_ship = []
for i, run in enumerate(run_no):
    Fn_p = calcFroudeNumber(v_ship[i], Lwl)
    Fn_ship.append( Fn_p )
    Re_p = calcReynoldsNumber(v_ship[i], Lwl, VISCOSITY_SEAWATER_12C)
    Re_ship.append( Re_p )
    Cf_p = calcCoeffFriction(Re_p)
    Cf_ship.append( Cf_p )
    Ct_p = Cw[i] + (1+form_factor)*Cf_p
    Ct_ship.append( Ct_p )
    
print("\nThe Froude Number of the Ship = \n", Fn_ship)
print("\nThe Reynold Number of the Ship = \n", Re_ship)
print("\nThe Coefficient of Frictional Resistance of the Ship = \n", Cf_ship)
print("\nThe Coefficient of Total Resistance of the Ship = \n", Ct_ship)


# Resistance Calculation of Scaled Measurements from Model to Prototype
R_total_ship = []
for i, run in enumerate(run_no):
    R_t_p = Ct_ship[i] * 0.5 * RHO_SEAWATER_12C * v_ship[i]**2 * S_wet_ship
    R_total_ship.append(R_t_p)
    
print("\nThe scaled total Resistance of the real Ship (in [N]) = \n", R_total_ship)

plt.title("Weerstandskromme van de Labrax")
plt.scatter(v_ship, [r/1000 for r in R_total_ship], c='orange', marker='o')
plt.plot(v_ship, [r/1000 for r in R_total_ship], label="Weerstandskromme uit sleeptankproef-meetdata")
plt.xlabel("geschaalde scheepssnelheid $v_s$ (in $[m/s]$)")
plt.ylabel("geschaalde scheepsweerstand $R_{total}$ (in $[kN]$)")
plt.grid(True)
plt.legend()
plt.show()

print("\n------- END RUN -------\n")