# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:14:03 2025

@author: fabio
"""
import numpy as np

D = 3.4 #m
Spoedverhouding = 1
rho_zee = 1024 #kg/m3
viscositeit = 1.3453 * 10**-6 #m2/s
g = 9.81 #m/s2

v1 = (10.5 * 1.852)/3.6 #m/s
v2 = (3 *1.852)/3.6 #m/s

n1 = 140 / 60 #rps
n2 = 80 / 60 #rps

kt1 = 0.22
kq1 = 0.365/10
ren1 = 0.56

kt2 = 0.33 
kq2 = 0.0505
ren2 = 0.305

def instroom(t, v):
 ve = v*(1-t)
 return ve

t = 0.15

ve1 = instroom(t, v1)

ve2 = instroom(t, v2)

def advrat(ve, d, n):
    j = ve/(d*n)
    return j

j1 = advrat(ve1, D, n1)

j2 =  advrat(ve2, D, n2)
print(j1)
print(j2)
def ktf(kt, n):
    T=kt*(rho_zee*(D**4)*(n**2))
    return T

def kqf(kq, n):
    q=kq*(rho_zee*(D**5)*(n**2))
    return q

T1 = ktf(kt1, n1)
T2 = ktf(kt2, n2)

q1 = kqf(kq1, n1)
q2 = kqf(kq2, n2)

def rendement(T, ve, q, n):
    ren = (T*ve)/(q*2*np.pi*n)
    return ren

#ren1 = rendement(T1, ve1, q1, n1)
#ren2 = rendement(T2, ve2, q2, n2)