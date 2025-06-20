# -*- coding: UTF-8 -*-
"""
Created by Xiaozhe Wang
        based on Matlab/simulink code of dr. P.deVos
        modified by E.Ulijn
TU Delft 3ME / MTT / SDPO / ME
Version 5.4h
initial release 19-March-2019

History:

20220628: EU: modified plots
20240209: EU: modified parameters for Labrax i.s.o. Visserskotter
20240209: EU: modified function names
20240212: EU: created EE function for electrical engine
              modified plot limits/titles
20240318: EU: modified Elec Gensets
20240322: EU: cleanup
20240411: EU: altered PowerPlant(anti-causal)
"""

from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import BDF, solve_ivp
from scipy.interpolate import interp1d as ip
from scipy.interpolate import interp1d
import math

#Initial experiment
Total_time = 36000 #[s] Total simulation time
X_iv_t_control = np.array([  0,  0.1*Total_time,  0.1*Total_time + 60,  0.2*Total_time, 0.2*Total_time + 60, 0.5*Total_time, 0.6*Total_time, 0.7*Total_time, 0.7*Total_time + 60, 0.8*Total_time, 0.8*Total_time + 60, Total_time]) #[s] setting time point
X_fs_unlim    =        np.array([1.0,            1,                     0.9,             0.9,                 0.2,            0.2,            0.2,            0.2,                 0.5,            0.5,                   1,           1]) #[-] fuel rack factor, 1 is maximum
Y_iv_t_control = np.array([  0,  0.1*Total_time,  0.1*Total_time + 60,  0.2*Total_time, 0.2*Total_time + 60, 0.5*Total_time, 0.6*Total_time, 0.7*Total_time, 0.7*Total_time + 60, 0.8*Total_time, 0.8*Total_time + 60, Total_time]) #[s] setting time point
Y_df    =        np.array([  1,               1,                    1,               1,                   1,              1,              1,              1,                   1,              1,                   1,          1]) #[-] disturbance factor

X_fs = np.zeros(len(X_fs_unlim))

for i in range(0, len(X_fs_unlim)):
    X_fs[i] = min(1, X_fs_unlim[i])
    if X_fs[i] < 0.2:
        X_fs[i] = 0

#Fuel Properties
LHV = 42700 #[kJ/kg]  Lower Heating Value
print('fuel properties loaded')

#Physical constant
rho_sea = 1025 				#[kg/m^3] density of sea water
print('water properties loaded')

#Ship Data
w = 0.3 					# [-] wake factor 
#c1 = 10000 					#[-] resistance coefficient of R = c1 * (v_s ^ 2)
thrust_factor = 0.15 		#[-] thrust deduction factor uit HM sheet Jasper
m_ship = 8000000 			#[kg] ship mass
print('ship data loaded')

#Propeller Data
D_p = 3.4 					#[m] propeller diameter
K_Ta = -0.09970875640677149 #[-] K_T factor a
K_Tb = -0.17206740568017384 #[-] K_T factor b
K_Tc = 0.36074407966305094 #[-] K_T factor c
K_Qa = -0.016753575062606815  #[-] K_Q factor a
K_Qb = -0.006573389455185741 #[-] K_Q factor b
K_Qc = 0.048660873932385294 #[-] K_Q factor c
eta_R = 1.00 				#[-] relative rotative efficiency
print('propellor data loaded')

#Internal Combustion Engine Data
m_fuelnom = .25 			#[g] nominal fuel injection
i = 6 						#[-] number of cylinders
k_es = 2 					#[-] k-factor for engines based on nr.of strokes per cycle
eta_gen = 0.97				# Generator set efficiency from P. de Vos
#eta_ICE = 0.38				# model P. de Vos
n_eng_nominal = 1800/60     # model P. de Vos
eta_comb = 1
print('ICE data loaded') 

#PowerPlant Data
P_aux = 0
eta_EPD = 0.95				# rendement Electric Power Distribution from P.de Vos
k_gensets = 4				# 
Q_loss_comb = 0
eta_td = 0.52
print('PowerPlant data loaded')

# Electrical motor data
# Electro Adda W40 LX6
eta_EM   = 0.97 			# redement Electro Motor from P. de Vos
P_EM_MAX = 650000 			# maximum power from Zuidgeest B.1.3
U_EM_nom = 660				# [V]
I_EM_nom = 984				# [A]
k_EM	 = 2				# 2 electr. motors
print('engine data loaded')

P_EM_nom = (U_EM_nom*I_EM_nom*k_EM)/eta_EM

P_EPD_nom = P_EM_nom/eta_EPD

P_ICE_nominal = P_EPD_nom/eta_gen

#Gearbox Data
i_gb = 6.369 				#[-] gearbox ratio
eta_TRM = 0.95 				# [-] transmission efficiency
I_tot = 200 				# [kg*m^2] total mass of inertia of propulsion system
print('gearbox data loaded')

#Initial Values
v_s0 = 8.5 * 0.5144444444	#[m/s] initial ship speed from verslag S.F.K.Zuidgeest
n_e0 = 900/60 				#[rps] Nominal engine speed in rotations from B.1.3. verslag S.F.K.Zuidgeest 
n_p0 = n_e0 / i_gb 			#[rps] initial rpm
s0 = 0 						#[m] travel distance
FC0 = 0 					#[g] fuel consumed
P_B0 = 2* 650 				#[kW] Nominal electric engine  power from Zuidgeest
M_B0 = P_B0 * 1000 / (2 * pi * n_e0) #([P_b*1000/(2*pi*n_eng_nom)])
ME0 = M_B0
MP0 = M_B0 * i_gb * eta_TRM

#Sub function
#Look-up fuelrack
def calc_X_fs_set(t, X_iv_t, Xfs):
    X_fs_set = np.interp(t, X_iv_t, Xfs)
    return X_fs_set

#Look-up disturbance
def calc_Y_df_set(t, Y_iv_t, Y_df):
    Y_df_set = np.interp(t, Y_iv_t_control, Y_df)
    return Y_df_set

#Advance Ratio
def calc_AdvanceRatio(v_s, n_p):
    v_a = v_s * (1 - w)
    J = v_a / (n_p * D_p)
    return [v_a, J]

#Propeller Thrust
def calc_PropThrust(v_s, n_p):
    K_T = K_Ta*(calc_AdvanceRatio(v_s, n_p)[1]**2) + K_Tb*calc_AdvanceRatio(v_s, n_p)[1] + K_Tc
    F_prop = K_T * (n_p ** 2) * rho_sea * (D_p ** 4)
    return [K_T, F_prop]

#Ship Resistance
# reynoldsmodel -> Cfmodel -> k -> Ctmodel -> Cwmodel -> Cwschip -> reynoldsschip -> Cfschip -> Ctschip -> Rtschip
def reynoldsgetalBerekening(V, L, vis):
    Re = (V*L)/vis
    return Re

def calcFroudeNumber(v_, L_):
    Fn_ = v_ / np.sqrt(9.81*L_)
    return Fn_

def cfBerekening(Re):
    Cf = 0.075/((np.log10(Re) - 2)**2)
    return Cf

def ctBerekening(R_tot, rho_water, v_model, S_model):
    Ct = R_tot/(0.5*rho_water*(v_model**2)*S_model)
    return Ct

def cwBerekening(Ct, Cf, k):
    Cw = Ct - (Cf * (1 + k))
    return Cw

# reynoldsmodel -> Cfmodel -> k -> Ctmodel -> Cwmodel -> Cwschip -> reynoldsschip -> Cfschip -> Ctschip -> Rtschip
alpha = 50
V_metingen = np.array([0.0973, 0.1968, 0.2957, 0.3948, 0.4941, 0.5934, 0.6925, 0.8904, 0.9896, 1.1880]) - 0.0016 # 0.7916, 
R_tot_model = np.array([0.8984, 1.0203, 1.1544, 1.3701, 1.6332, 1.9424, 2.3537, 3.4843, 4.8205, 9.5683]) - 0.8402 # 2.1885, 
L_schip = 117.7
L_model = L_schip/alpha
S_schip = 2661.0
S_model = S_schip/(alpha**2)
vis_sleeptank = 1.0621*(10**(-6))
vis_schip = 1.28324*(10**(-6))
rho_model = 998.6536
rho_schip = 1026.6376

Re_model = reynoldsgetalBerekening(V_metingen, L_model, vis_sleeptank)

Cf_model = cfBerekening(Re_model)
Ct_model = ctBerekening(R_tot_model, rho_model, V_metingen, S_model)

Fn_model = calcFroudeNumber(V_metingen, L_schip/alpha)

Ctm_vs_Cfm = []
Fnm_vs_Cfm = []
for i, ctm in enumerate(Ct_model):
    if Fn_model[i] > 0.1 and Fn_model[i] < 0.2:
        Ctm_vs_Cfm.append( Ct_model[i]/Cf_model[i] ) 
        Fnm_vs_Cfm.append( (Fn_model[i]**4)/Cf_model[i] )
plt.title("Prohaska Plot")
plt.scatter(Fnm_vs_Cfm, Ctm_vs_Cfm)
plt.xlabel("$Fn^4/C_f$")
plt.ylabel("$C_t/C_f$")
plt.grid(True)
plt.show()
interpol_formfactor = ip(Fnm_vs_Cfm, Ctm_vs_Cfm, kind='linear', fill_value='extrapolate')
form_factor = interpol_formfactor(0.0) - 1 # Volgens methode dat de functie de ordinaat bij 1+k intersect.
print("\nForm Factor k of Model and Prototype = \n", form_factor)

k = form_factor
Cw_model = cwBerekening(Ct_model, Cf_model, k)
Cw_schip = Cw_model

def cBerekening(cw, cf, k, v_s):
    cw_func = ip(V_metingen, Cw_model, bounds_error=False, fill_value='extrapolate')
    c = cw_func(v_s) + cf*(1 + k)
    return c

def calc_ShipResistance(v_s, Y_df_set):
    Re_schip = reynoldsgetalBerekening(v_s, L_schip, vis_schip)
    cf_schip = cfBerekening(Re_schip)
    c1 = cBerekening(Cw_schip, cf_schip, k, v_s)
    R = 0.5*(Y_df_set * c1 * (v_s**2))*rho_schip*S_schip
    R =11483.74143794+(2800.40244106*v_s)+(1702.4357436*(v_s**2))
    R_sp = R / (1 - thrust_factor)
    return [R, R_sp]

#Ship Translational Dynamics
def ShipTransDynamics(v_s, n_p, Y_df_set):
    dv_sdt = (calc_PropThrust(v_s, n_p)[1] - calc_ShipResistance(v_s, Y_df_set)[1]) / m_ship
    dsdt = v_s
    return [dv_sdt, dsdt]

#Propeller Torque
def Prop_torque(v_s, n_p):
    K_Q = K_Qa*(calc_AdvanceRatio(v_s, n_p)[1]**2) + K_Qb*calc_AdvanceRatio(v_s, n_p)[1] + K_Qc
    Q = K_Q * (n_p ** 2) * rho_sea * (D_p ** 5)
    M_prop = Q / eta_R
    return [K_Q, Q, M_prop]

#Internal Combustion Engine
def ICE(P_ICE, n_eng):
    M_B = P_ICE / (2 * pi * n_eng)
    fire_freq = n_eng* i /k_es
    Q_loss_cooling = 795.33 + (3181.333 * (P_ICE / P_ICE_nominal))
    W_loss_mech = 296.29 + (691.38 * (n_eng / n_eng_nominal))
    W_e_p = (M_B * fire_freq)
    W_e = W_e_p + W_loss_mech
    eta_mech = W_e_p/W_e
    Q_f_p = (W_e / eta_td)
    Q_f = Q_f_p + Q_loss_cooling
    eta_ICE = Q_f_p/Q_f
    m_f = Q_f / LHV #dFCdt    
    eta_tot = eta_mech*eta_ICE*eta_comb*eta_td#(W_e/Q_f)
    return m_f, eta_ICE, eta_mech, eta_tot
    
#Electric Motor
def Electric_Motors(X_fs_set, n_EM):
    P_EM = X_fs_set* U_EM_nom * k_EM * I_EM_nom
    P2= P_EM* eta_EM # Mechanisch vermogen 
    M_EM = P2 / (n_EM * 2 * pi)
    return [P_EM, M_EM]

#Power Plant
def PowerPlant(P_EM):
    P_elec_gen =  (P_EM + P_aux)/eta_EPD 		# ElectricPowerDistributionSystem
    P_ICE = ((P_elec_gen /eta_gen) / k_gensets)	# 4GenSets
    
    m_f = ICE(P_ICE, n_eng_nominal)[0]
    m_flux = m_f*n_eng_nominal*i/k_es
    out_m_flux = m_flux * k_gensets
    return (out_m_flux)


# GenSets
def Gensets(P_elec_gen):
    P_ICE = P_elec_gen/eta_gen
    P_ICE = P_ICE / k_gensets
    return P_ICE

# Drive system block
#Shaft Rotational Dynamics
def SRD(M_B, M_P):
    delta_M=M_B*i_gb*eta_TRM - M_P
    dn_pdt = delta_M/(2*pi*I_tot) 
    
    return [dn_pdt]

# Main program function
# y[0]-v_s  
# y[1]-n_p 
# y[2]-s 
# y[3]-FC 

def main_simulation(t,y):
    X_fs_set = calc_X_fs_set(t, X_iv_t_control, X_fs)
    Y_df_set = calc_Y_df_set(t, Y_iv_t_control, Y_df)
    dv_sdt = ShipTransDynamics(y[0], y[1], Y_df_set)[0]	#Ship Translational Dynamics
    dsdt = ShipTransDynamics(y[0], y[1], Y_df_set)[1]	#Ship Translational Dynamics
    n_EM = y[1]*i_gb
    P_EM, M_EM = Electric_Motors(X_fs_set, n_EM)
    dFCdt=PowerPlant(P_EM)
    M_P = Prop_torque(y[0], y[1])[2]
    dn_pdt = SRD(M_EM, M_P)[0]
    return[dv_sdt, dn_pdt, dsdt, dFCdt]
    
    

#ODE solver
sol = solve_ivp(main_simulation, [0, Total_time], [v_s0, n_p0, s0, FC0], method='BDF')

#Simulation output
v_s, n_p, s, FC = sol.y
FC = k_gensets  * FC 									# aantal generatoren.
X = calc_X_fs_set(sol.t, X_iv_t_control, X_fs)
Y = calc_Y_df_set(sol.t, Y_iv_t_control, Y_df)
R = calc_ShipResistance(v_s, Y)[0]
J = calc_AdvanceRatio(v_s, n_p)[1]
n_EM = n_p * i_gb
P_EM = Electric_Motors(X, n_EM)[0]
M_EM = Electric_Motors(X, n_EM)[1]
M_prop = Prop_torque(v_s, n_p)[2]
P_P = M_prop * 2 * pi * n_p
P_B = Gensets(P_EM)
R_sp = calc_ShipResistance(v_s, Y)[1]
F_prop = calc_PropThrust(v_s, n_p)[1]
K_T = calc_PropThrust(v_s, n_p)[0]
K_Q = Prop_torque(v_s, n_p)[0]
v_a = calc_AdvanceRatio(v_s, n_p)[0]
P_T = F_prop * v_a
Q = Prop_torque(v_s, n_p)[1]
P_O = 2 * pi * Q * n_p
Q_f, eta_ICE_plot, eta_mech_plot, eta_tot_plot = ICE(P_B, n_eng_nominal)
eta_hull = (1 - w) /(1 - P_T)
eta_O = P_T / P_O
eta_EM = eta_EM * np.ones( len(sol.t))

print("Time point length:", len(sol.t))

#Plot Figure
width = 0.8 #Plot line width setting

plt.figure(figsize=(9,6))
plt.subplot(4, 1, 1)
plt.xlim(0,Total_time)
plt.ylim(0,15)
plt.plot(sol.t, (v_s*3.6)/1.852, linewidth=width)
plt.title('Ship Propulsion Output')
plt.xlabel('Time [s]')
plt.ylabel('Ship speed [m/s]')
plt.grid(True)
plt.subplot(4, 1, 2)
plt.xlim(0,Total_time)
plt.plot(sol.t, s, linewidth=width)
plt.xlabel('Time [s]')
plt.ylabel('Distance travelled [m]')
plt.grid(True)
plt.subplot(4, 1, 3)
plt.xlim(0,Total_time)
plt.plot(sol.t, FC/1000, linewidth=width)
plt.xlabel('Time [s]')
plt.ylabel('Fuel consumed [kg]')
plt.grid(True)
plt.subplot(4, 1, 4)
plt.xlim(0,Total_time)
plt.plot(sol.t, X, linewidth=width)
plt.xlabel('Time [s]')
plt.ylim(0,1)
plt.ylabel('Fuel rack [%]')
plt.grid(True)
plt.tight_layout()
plt.savefig('start_fig01.jpg')
plt.show()

plt.plot(P_B, eta_ICE_plot)
plt.plot(P_B, eta_mech_plot)
plt.plot(P_B, [eta_td]*len(P_B))
plt.plot(P_B, [eta_comb]*len(P_B))
plt.show()

plt.plot((v_s*3.6)/1.852, R_sp)

