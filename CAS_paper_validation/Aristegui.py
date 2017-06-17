#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:39:30 2017
Validation of SAFE routines with spectral elements
water filled copper pipe in water
http://dx.doi.org/10.1016/S0041-624X(01)00064-6
@author: michal
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
plt.style.use('/home/michal/Dropbox/plot_templates/jsv.mplstyle')

from context import axisafe
#%%
lame_1, lame_2 = axisafe.misc.cLcS2lame(4759, 2325, 8933)
copper = [lame_1, lame_2, 8933, 0.00]
water = [2.25e9, 1000, 0]
h_long, h_short, PML_order, R_s, R_l, R_sl = \
        axisafe.mesh.suggest_PML_parameters(0.0075, 8, copper, water, 1e3, 4e5, att=1)


sets = [['water', 0.0, 0.0068, water, 'ALAX6'], 
        ['copper', 0.0068, 0.0007, copper, 'SLAX6'], 
        ['water_a', 0.0075, np.round(h_short, 4), water, 'ALAX6_PML', PML_order]]
        
f = np.linspace(1e3, 4e5, 200, True)
#%%
mesh0 = axisafe.mesh.Mesh()
mesh0.create(sets, f[-1], PML='water_a', 
             PML_props=[0.0075, np.round(h_short, 4), 6 + 7j])
mesh0.assemble_matrices(n=0)

pipe0 = axisafe.solver.WaveElementAxisym(mesh0)
pipe0.solve(f)
pipe0.energy_ratio()
pipe0.k_propagating(100, 0.9)
pipe0.energy_velocity()
k_0 = np.copy(pipe0.k_ready)
#%%
mats = []
for i in range(1, 8):
    temp = np.genfromtxt('reference/aristegui_submerged/d'+str(i)+'.csv', 
                         delimiter=',')
    mats.append(temp)
mats_att = []
for i in range(1, 6):
    temp = np.genfromtxt('reference/aristegui_submerged/attenuation/L0'+str(i)+'.csv', 
                         delimiter=',')
    mats_att.append(temp)
k_00 = pipe0.k[:, pipe0.propagating_indices]
k_00[k_00.imag > 10] = np.nan + 1j*np.nan
k_00[k_00.imag < -2000/8.7] = np.nan + 1j*np.nan
k_00 = k_00[:, np.argsort(k_0[-1])]

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(7.48, 3))

for i in range(7):
    arist, = [ax1.plot(mats[i][:, 0], mats[i][:, 1], 'o', mec='tab:blue', mfc='none', ms=4, 
                 markevery=2)]
mymodel, = [ax1.plot(f[1:]/1e6, 
         1e-3*np.diff(2*np.pi*f.reshape(-1, 1), axis=0)/np.diff(k_00, axis=0), '-',
         c='tab:orange', lw=2)]

for i in range(5):
    ax2.plot(mats_att[i][:, 0], mats_att[i][:, 1], 'o', mec='tab:blue', mfc='none', 
             ms=4, markevery=2)
ax2.plot(f/1e6, -20*np.log10(np.e)*k_00.imag, '-',c='tab:orange', lw=2)

ax1.set_xlim([0, .4])
ax1.set_ylim([0, 5])
ax2.set_ylim([0, 10])
ax2.set_xlabel('frequency in MHz')
ax1.set_xlabel('frequency in MHz')
leg = ax1.legend([arist[-1], mymodel[-1]], ['from Aristegui et al. [45]', 
                  'present model'], loc=2, fontsize='x-small', frameon=True)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(0.8)
ax1.set_ylabel(' group velocity in km/s')
ax2.set_ylabel(' attenuation in dB/m')
ax1.grid(lw=0.)
ax2.grid(lw=0.)

plt.tight_layout()
#fig.savefig('output_figures/copper_pipe.pdf', transparent=True, dpi=600)

#%%
