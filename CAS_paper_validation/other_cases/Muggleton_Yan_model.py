#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:06:00 2017

An implementation fo the model fro mMuggleton and Yan (2013)

@author: Michal Kalkowski
"""

from __future__ import print_function
import numpy as np
from scipy.special import hankel2

def KG_to_lame(K, G):
    lame1 = K - 2*G/3
    lame2 = G
    return lame1, lame2

def cLcS2lame(cl, ct, rho):
    lame_1 = (cl**2 - 2*ct**2)*rho
    lame_2 = ct**2*rho
    return lame_1, lame_2

def dH2n_dz(n, z):
    return 0.5*(hankel2(n - 1, z) - hankel2(n + 1, z)).squeeze()

def d2H2n_dz2(n, z):
    return 0.5*(dH2n_dz(n - 1, z) - dH2n_dz(n + 1, z)).squeeze()

def root_sign(root):
    if root.real >= root.imag:
        return root
    else:
        return -root

def k1_2004(K_f, a, E, h, w, rho, lame1_soil, lame2_soil, rho_soil, kf):
    # recursive function for calculating k1 based on a guess of k1 based on eq. (36)
    

    def k1_m(guess, K_f, a, E, h, w, rho, lame1_soil, lame2_soil, rho_soil, kf):
        
        kd = w*(rho_soil/(lame1_soil + 2*lame2_soil))**0.5
        kds_r = root_sign((kd**2 - guess**2)**0.5)
        if lame2_soil == 0:
            waves_radiating = [[kd, kds_r]]
        else:
            kr = w*(rho_soil/lame2_soil)**0.5
            krs_r = root_sign((kr**2 - guess**2)**0.5)
            waves_radiating = [[kd, kds_r], [kr, krs_r]]
        
        fluid_stiffness = 2*K_f/a
        pipe_wall_stiffness = E*h/a**2
        pipe_wall_mass = -w**2*rho*h
        
        Z_fluid = -1j*fluid_stiffness/w
        Z_pipe = pipe_wall_mass/(1j*w) - 1j*pipe_wall_stiffness/w
        Z_rad = 0
        for wave in waves_radiating:
            Z_rad += -1j*rho_soil*w/wave[1]*\
                                hankel2(0, wave[1]*a)/(dH2n_dz(0, wave[1]*a))
        
        k1 = kf*(1 + Z_fluid/(Z_pipe + Z_rad))**0.5
        return k1
    fluid_stiffness = 2*K_f/a
    pipe_wall_stiffness = E*h/a**2
    pipe_wall_mass = -w**2*rho*h
    
    Z_fluid = -1j*fluid_stiffness/w
    Z_pipe = pipe_wall_mass/(1j*w) - 1j*pipe_wall_stiffness/w
    # approximation to k1 at low freq (free)
    k1_approx = kf*(1 + Z_fluid/Z_pipe)**0.5
    
    k1 = []
    for i in range(len(w)):
        ks = []
        res = 1
        # see when it converges
        while res > 1e-10:
            if len(ks) == 0:
                if i == 0:
                    k1_guess = k1_approx[i]
                else:
                    k1_guess = k1[-1]
                k_loc = k1_m(k1_guess, K_f, a, E, h, w[i], rho, lame1_soil, lame2_soil, rho_soil, kf[i])
                ks.append(k_loc)
            else:
                k_loc = k1_m(ks[-1], K_f, a, E, h, w[i], rho, lame1_soil, lame2_soil, rho_soil, kf[i])
                ks.append(k_loc)
            if len(ks) > 1:
                res = abs(k_loc - ks[-2])
            else:
                res = 1
        k1.append(ks[-1])
    k1_final = np.array(k1)
    return k1_final
    
def k1_my(K_f, a, E, h, w, rho, lame1_soil, lame2_soil, rho_soil, kf):
    # recursive function for calculating k1 based on a guess of k1 based on eq. (36)
    def k1_m(guess, K_f, a, E, h, w, rho, lame1_soil, lame2_soil, rho_soil, kf):
        if lame1_soil == 0:
            kd = 0
        else:
            kd = w*(rho_soil/(lame1_soil + 2*lame2_soil))**0.5
        if lame2_soil == 0:
            kr = 0
        else:
            kr = w*(rho_soil/lame2_soil)**0.5
        kds_r = root_sign((kd**2 - guess**2)**0.5)
        krs_r = root_sign((kr**2 - guess**2)**0.5)
        cd = w/kd
        z_rad = -1j*rho_soil*cd*kd/kds_r*hankel2(0, kds_r*a)/(dH2n_dz(0, kds_r*a))
        fluid_stiffness = 2*K_f/a
        pipe_wall_stiffness = E*h/a**2
        pipe_wall_mass = -w**2*rho*h
        compr_wave_imp = 1j*w*(lame1_soil/(lame1_soil + 2*lame2_soil))*(1 - 2*guess**2/kr**2)*z_rad
        compr_rad_term = 2*lame2_soil/a*(1 - 2*guess**2/kr**2)*kds_r*a*d2H2n_dz2(0, kds_r*a)/dH2n_dz(0, kds_r*a)
        shear_rad_term = 4*lame2_soil/a*guess**2/kr**2*krs_r*a*dH2n_dz(1, krs_r*a)/hankel2(1, krs_r*a)
        k1 = kf*(1 + fluid_stiffness/(pipe_wall_stiffness + pipe_wall_mass + compr_wave_imp - 
                                  compr_rad_term - shear_rad_term))**0.5
        return k1
    # approximation to k1 at low freq
    k1_eq38 = kf*(1 + 2*K_f/a/(E*h/a**2 + 2*lame2_soil/a))**0.5
    
    k1 = []
    for i in range(len(w)):
        ks = []
        res = 1
        # see when it converges
        while res > 1e-10:
            if len(ks) == 0:
                if i == 0:
                    k1_guess = k1_eq38[i]
                else:
                    k1_guess = k1[-1]
                k_loc = k1_m(k1_guess, K_f, a, E, h, w[i], rho, lame1_soil, lame2_soil, rho_soil, kf[i])
                ks.append(k_loc)
            else:
                k_loc = k1_m(ks[-1], K_f, a, E, h, w[i], rho, lame1_soil, lame2_soil, rho_soil, kf[i])
                ks.append(k_loc)
            if len(ks) > 1:
                res = abs(k_loc - ks[-2])
            else:
                res = 1
        k1.append(ks[-1])
    k1_final = np.array(k1)
    return k1_final

def k2_my(K_f, a, E, nu, h, w, rho, lame1_soil, lame2_soil, rho_soil, kf):
    # recursive function for calculating k1 based on a guess of k1 based on eq. (40)
    def k2_m(guess, K_f, a, E, nu, h, w, rho, lame1_soil, lame2_soil, rho_soil, kf):
        kL = (w**2 *rho*(1 - nu**2)/E)**0.5
        kd = w*(rho_soil/(lame1_soil + 2*lame2_soil))**0.5
        kr = w*(rho_soil/lame2_soil)**0.5
        kds_r = root_sign((kd**2 - guess**2)**0.5)
        krs_r = root_sign((kr**2 - guess**2)**0.5)
        cd = w/kd
        z_rad = -1j*rho_soil*cd*kd/kds_r*hankel2(0, kds_r*a)/(dH2n_dz(0, kds_r*a))
        fluid_stiffness = 2*K_f/a
        pipe_wall_stiffness = E*h/a**2
        pipe_wall_mass = -w**2*rho*h
        compr_wave_imp = 1j*w*(lame1_soil/(lame1_soil + 2*lame2_soil))*(1 - 2*guess**2/kr**2)*z_rad
        compr_rad_term = 2*lame2_soil/a*(1 - 2*guess**2/kr**2)*kds_r*a*d2H2n_dz2(0, kds_r*a)/dH2n_dz(0, kds_r*a)
        shear_rad_term = 4*lame2_soil/a*guess**2/kr**2*krs_r*a*dH2n_dz(1, krs_r*a)/hankel2(1, krs_r*a)
        k2 = kL*(1 + nu**2/(1 - nu**2)*pipe_wall_stiffness/(pipe_wall_stiffness + fluid_stiffness + \
                                                            pipe_wall_mass + compr_wave_imp - \
                                                            compr_rad_term - shear_rad_term))**0.5
        return k2
    # approximation to k1 at low freq
    kL = (w**2 *rho*(1 - nu**2)/E)**0.5
    k2_eq42 = kL*(1 + nu**2/(1 - nu**2)*E*h/a**2/(E*h/a**2 + 2*K_f/a + 2*lame2_soil/a))**0.5
    
    k2 = []
    for i in range(len(w)):
        ks = []
        res = 1
        # see when it converges
        while res > 1e-10:
            if len(ks) == 0:
                if i == 0:
                    k2_guess = k2_eq42[i]
                else:
                    k2_guess = k2[-1]
                k_loc = k2_m(k2_guess, K_f, a, E, nu, h, w[i], rho, lame1_soil, lame2_soil, rho_soil, kf[i])
                ks.append(k_loc)
            else:
                k_loc = k2_m(ks[-1], K_f, a, E, nu, h, w[i], rho, lame1_soil, lame2_soil, rho_soil, kf[i])
                ks.append(k_loc)
            if len(ks) > 1:
                res = abs(k_loc - ks[-2])
            else:
                res = 1
        k2.append(ks[-1])
    k2_final = np.array(k2)
    return k2_final