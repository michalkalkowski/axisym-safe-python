#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:39:30 2017
Validation of SAFE routines with spectral elements
Torsional waves in a cast iron pipe buried in sand
@author: michal
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
plt.style.use('/home/michal/Dropbox/plot_templates/jsv.mplstyle')

from context import axisafe
import Muggleton_Yan_model as Jens_model

#%%
lame_1, lame_2 = axisafe.misc.young2lame(2e9, 0.4)
mdpe = [lame_1, lame_2, 900, 0.06]
water = [2.25e9, 1000, 0]
h_long, h_short, PML_order, R_s, R_l, R_sl = \
        axisafe.mesh.suggest_PML_parameters(0.09, 3, mdpe, water, 1, 1e3, att=1)

sets = [['water', 0.0, 0.079, water, 'ALAX6'], 
        ['mdpe', 0.079, 0.011, mdpe, 'SLAX6'], 
        ['water_a', 0.09, np.round(h_short, 4), water, 'ALAX6_PML', PML_order]]
f = np.linspace(1, 1000, 100, True)
#%%
mesh0 = axisafe.mesh.Mesh()
mesh0.create(sets, f[-1], PML='water_a', PML_props=[0.09, np.round(h_short, 4), 6 + 7j])
mesh0.assemble_matrices(n=0)

pipe0 = axisafe.solver.WaveElementAxisym(mesh0)
pipe0.solve(f)
pipe0.energy_ratio()
pipe0.k_propagating(15, 0.8)
k_0 = np.copy(pipe0.k_ready)
#%%
sets = [['water', 0.0, 0.079, water, 'ALAX6'], 
        ['mdpe', 0.079, 0.011, mdpe, 'SLAX6']]

mesh1 = axisafe.mesh.Mesh()
mesh1.create(sets, f[-1])
mesh1.assemble_matrices(n=0)

pipe1 = axisafe.solver.WaveElementAxisym(mesh1)
pipe1.solve(f)
pipe1.energy_ratio()
pipe1.k_propagating(15, 1)
k_1 = np.copy(pipe1.k_ready)
#%%
#%% compute analytical results from Muggleton Yan model
kf = 2*np.pi*f/(water[0]/water[1])**0.5
                
k1_jen_water = Jens_model.k1_2004(water[0], 0.0845, 2e9*(1 + 0.06j), 0.011, 
                             2*np.pi*f, 900, water[0], 0, 1000, kf)

#%% From Jen Muggleton's torsional waves paper in JSV 2016
Jens = loadmat('reference/muggleton_2004/JMM_data.mat')
#%%   5
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(7.48, 3))
safe_water, = [ax1.plot(f, k_0[:, 0].real, '.-')]
safe_air, = [ax1.plot(f, k_1[:, 0].real, '.-')]
ax1.set_prop_cycle(None)
jen_water, = ax1.plot(Jens['inwater_exp'][:, 0], Jens['inwater_exp'][:, 1].real, '-')
jen_air, = ax1.plot(Jens['inair_exp'][:, 0], Jens['inair_exp'][:, 1].real, '-')
ax2.set_prop_cycle(None)
#analytic, = ax1.plot(  f, k1_jen_water.real, ':')

ax2.plot(f, -20*np.log10(np.e)*k_0[:, 0].imag, '.-')
ax2.plot(f, -20*np.log10(np.e)*k_1[:, 0].imag, '.-')
ax2.set_prop_cycle(None)
ax2.plot(Jens['inwater_exp'][:, 0], 
         -20*np.log10(np.e)*Jens['inwater_exp'][:, 1].imag, '-')
ax2.plot(Jens['inair_exp'][:, 0], 
         -20*np.log10(np.e)*Jens['inair_exp'][:, 1].imag, '-')
ax2.set_prop_cycle(None)
#ax2.plot(f, -20*np.log10(np.e)*k1_jen_water.imag, ':')

ax1.legend([safe_water[0], safe_air[0], jen_water, jen_air], 
           ['SAFE in water', 'SAFE in air', 'exp in water', 
           'exp in air'], loc=2)
ax1.set_xlim([0,1000])
ax1.set_ylim([0, 35])
ax2.set_ylim([-10, 15])
ax2.set_xlabel('frequency in Hz')
ax1.set_xlabel('frequency in Hz')

ax1.set_ylabel(' real k in rad/m')
ax2.set_ylabel(' attenuation in dB/m')
ax1.grid(lw=0.)
ax2.grid(lw=0.)
plt.tight_layout()
#plt.savefig('output_figures/plastic_pipe_in_water.pdf', transparent=True, dpi=600)

#%% 
