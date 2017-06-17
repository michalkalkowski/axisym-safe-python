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
lame_1i, lame_2i = axisafe.misc.cLcS2lame(200, 100, 1500)
soil = [lame_1i, lame_2i, 1500, 0.]
water = [2.25e9, 1000, 0]

h_long, h_short, PML_order, R_s, R_l, R_sl = \
        axisafe.mesh.suggest_PML_parameters(0.09, 3, mdpe, soil, 1, 1e3, att=1)

sets = [['water', 0.0, 0.079, water, 'ALAX6'], 
        ['mdpe', 0.079, 0.011, mdpe, 'SLAX6'], 
        ['soil', 0.09, np.round(h_short, 4), soil, 'SLAX6_PML', PML_order]]

f = np.linspace(1, 1000, 50, True)
#%%
mesh0 = axisafe.mesh.Mesh()
mesh0.create(sets, f[-1], PML='soil', PML_props=[0.09, np.round(h_short, 4), 6 + 7j])
mesh0.assemble_matrices(n=0)

pipe0 = axisafe.solver.WaveElementAxisym(mesh0)
pipe0.solve(f)
pipe0.energy_ratio()
pipe0.k_propagating(15, 0.935)
k_0 = np.copy(pipe0.k_ready)

#%%
#%% compute analytical results from Muggleton Yan model
kf = 2*np.pi*f/(water[0]/water[1])**0.5
                
k1_jen_16 = Jens_model.k1_my(water[0], 0.0845, 1.6e9*(1 + 0.06j), 0.011, 
                             2*np.pi*f, 900, soil[0], soil[1], soil[2], kf)
k1_jen_20 = Jens_model.k1_my(water[0], 0.0845, 2e9*(1 + 0.06j), 0.011, 
                             2*np.pi*f, 900, soil[0], soil[1], soil[2], kf)

Jens_compact = loadmat('reference/yan_muggleton_2016/compact_coupling.mat')
#%% From Jen Muggleton's torsional waves paper in JSV 2016
Jens = np.genfromtxt('reference/muggleton_2002/Jens_exp_results.txt',
                      skip_header=1)
     
#%% 
#%% From Jen Muggleton's torsional waves paper in JSV 2016
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(7.48, 3))
safe2, = [ax1.plot(  f, k_0[:, 2].real, '.-')]
analytic16, = ax1.plot(Jens_compact['compact'][:, 0], Jens_compact['compact'][:, 1], '--')

ax2.plot(f, -20*np.log10(np.e)*k_0[:, 2].imag, '.-')
ax2.plot(Jens_compact['compact'][:, 0], -Jens_compact['compact'][:, 2], '--')
jen, = ax1.plot(Jens[:, 0], Jens[:, 1], lw=1)
ax2.plot(Jens[:, 0], -Jens[:, 2], lw=1)

ax1.legend([safe2[0], analytic16, jen], 
           ['SAFE ', 'analytical', 'experiment'], loc=2)
ax1.set_xlim([0,800])
ax1.set_ylim([0, 16])
ax2.set_ylim([-5, 45])
ax2.set_xlabel('frequency in Hz')
ax1.set_xlabel('frequency in Hz')

ax1.set_ylabel(' real k in rad/m')
ax2.set_ylabel(' attenuation in dB/m')
ax1.grid(lw=0.)
ax2.grid(lw=0.)
plt.tight_layout()
#plt.savefig('output_figures/plastic_pipe_in_soil.pdf', transparent=True, dpi=600)
