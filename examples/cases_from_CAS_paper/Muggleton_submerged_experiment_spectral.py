#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of SAFE routines with spectral elements
:Waves in a water-filled MDPE pipe submerged in water:
    
Computes a case found in:
Muggleton, J.M., Brennan, M.J., 2004. Leak noise propagation and attenuation in 
submerged plastic water pipes. Journal of Sound and Vibration 278, 527–537. 
doi:10.1016/j.jsv.2003.10.052
    
@author: Michal K Kalkowski, kalkowski.m@gmail.com
Copyright (c) 2017 Michal Kalkowski (MIT license)
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../mkk_style.mplstyle')

from context import axisafe
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
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(7.48, 3))
safe_water, = [ax1.plot(f, k_0[:, 0].real, '.-')]
safe_air, = [ax1.plot(f, k_1[:, 0].real, '.-')]

ax2.plot(f, -20*np.log10(np.e)*k_0[:, 0].imag, '.-')
ax2.plot(f, -20*np.log10(np.e)*k_1[:, 0].imag, '.-')

ax1.legend([safe_water[0], safe_air[0]], ['in water', 'in air'], loc=2)
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