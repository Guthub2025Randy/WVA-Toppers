# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:54:37 2025

@author: fabio & Daniël
"""
import numpy as np
import matplotlib.pyplot as plt

g = 9.81
rho = 1024
D_schroef = 3.4
t = 0.15

Q0 = -16.2201
T0 = -29.4130
I0 = 170 
n0 = 0.6572

v = (np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.6])*(1-t)**np.sqrt(50))
Q = (np.array([96.4461, 95.7446, 90.0793, 84.1381, 78.4771, 74.6180, 70.3828, 66.2057, 61.9759, 57.1878, 51.0588, 29.4138])-Q0)*50**2
T = (np.array([-22.2581, -22.6951, -23.3679, -24.0718, -24.4320, -24.8441, -25.2743, -25.6984, -26.0929, -26.5487, -26.9814, -28.5568])-T0)*50**3
n_rps = (np.array([14.9274, 14.9519, 14.8708, 14.8952, 14.9117, 14.8804, 14.8786, 14.8855, 14.8866, 14.8805, 14.8931, 14.8827])-n0)/np.sqrt(50)
n_rpm = n_rps * 60

I = (np.array([1160, 1180, 1130, 1080, 1030, 990, 965, 930, 890, 845, 810, 800]) - I0) * 10**-3

def regressionSecondDegreePolynomial(x_array_, a_, b_, c_):
    y_regression = a_*x_array_**2 + b_*x_array_ + c_
    return y_regression

J_proto = v / (n_rps * D_schroef)
print(J_proto)

Kt_proto = T / (rho*(D_schroef**4)*(n_rps**2))
print(Kt_proto)

Kq_proto = Q / (rho*(D_schroef**5)*(n_rps**2))
print(Kq_proto)

n_0_proto = (Kt_proto/Kq_proto) * (J_proto/2*np.pi)
print(n_0_proto)

# Approximation Fitting
a_Kt, b_Kt, c_Kt = np.polyfit(J_proto, Kt_proto, 2)
a_Kq, b_Kq, c_Kq = np.polyfit(J_proto, Kq_proto, 2)
print("Coefficients for second degree polynomial regression of J-Kt = ",a_Kt, b_Kt, c_Kt)
print("Coefficients for second degree polynomial regression of J-Kq = ",a_Kq, b_Kq, c_Kq)

Kt_proto_regression = regressionSecondDegreePolynomial(J_proto, a_Kt, b_Kt, c_Kt)
Kq_proto_regression = regressionSecondDegreePolynomial(J_proto, a_Kq, b_Kq, c_Kq)