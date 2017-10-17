#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:39:30 2017
Validation of SAFE routines with spectral elements (fluid PML)
:Waves in a titanum bar submerged in oil:
    
Comparison between axisafe and:
Hauke Gravenkamp, Carolin Birk, Chongmin Song, Numerical modeling of elastic waveguides coupled to infinite fluid media using exact boundary conditions, Computers & Structures, Volume 141, August 2014, Pages 36-45, ISSN 0045-7949, http://dx.doi.org/10.1016/j.compstruc.2014.05.010.
(http://www.sciencedirect.com/science/article/pii/S0045794914001242)

@author: Michal K Kalkowski, kalkowski.m@gmail.com
Copyright (c) 2017 Michal Kalkowski (MIT license)
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
plt.style.use('mkk_style.mplstyle')

from context import axisafe
#%%
lame_1, lame_2 = axisafe.misc.G_nu2lame(46.53e9, 0.302)
titanium = [lame_1, lame_2, 4460, 0.0]
oil = [2.634e9, 870, 0]

h_long, h_short, PML_order, R_s, R_l, R_sl = \
        axisafe.mesh.suggest_PML_parameters(0.001, 3, titanium, oil, 1e3, 3e6, att=1)

sets = [['titanium', 0.0, 0.001, titanium, 'SLAX6'], 
        ['oil', 0.001, np.round(h_short, 4), oil, 'ALAX6_PML', PML_order]]
        
f = np.linspace(1e3, 3e6, 100, True)
#%%
mesh0 = axisafe.mesh.Mesh()
mesh0.create(sets, f[-1], PML='oil', PML_props=[0.001, np.round(h_short, 4), 6 + 7j])

mesh0.assemble_matrices(n=0)

pipe0 = axisafe.solver.WaveElementAxisym(mesh0)

pipe0.solve(f)
pipe0.energy_ratio()
pipe0.k_propagating(350, 0.97)

k_0 = np.copy(pipe0.k_ready)
#%%
mesh1 = axisafe.mesh.Mesh()
mesh1.create(sets, f[-1], PML='oil', PML_props=[0.001, np.round(h_short, 4), 6 + 7j])

mesh1.assemble_matrices(n=1)

pipe1 = axisafe.solver.WaveElementAxisym(mesh1)


pipe1.solve(f)
pipe1.energy_ratio()
pipe1.k_propagating(350, 0.97)

k_1 = np.copy(pipe1.k_ready)
#%%
mesh2 = axisafe.mesh.Mesh()
mesh2.create(sets, f[-1], PML='oil', PML_props=[0.001, np.round(h_short, 4), 6 + 7j])

mesh2.assemble_matrices(n=2)

pipe2 = axisafe.solver.WaveElementAxisym(mesh2)
pipe2.solve(f)
pipe2.energy_ratio()
pipe2.k_propagating(350, 0.97)

k_2 = np.copy(pipe2.k_ready)

#%%
cp_HG = loadmat('reference/gravenkamp/cp.mat')
att_HG = loadmat('reference/gravenkamp/att.mat')
#%%
k = np.column_stack((k_0, k_1, k_2))
cp = 2e-3*np.pi*f.reshape(-1, 1)/k.real
att = -20*np.log10(np.e)*k.imag
#att[cp > 10] = np.nan
cp[abs(att) > 2313] = np.nan
att[abs(att) > 2313] = np.nan
#cp[(att) < -1e-3] = np.nan
cp[np.isinf(cp)] = np.nan
att[att == 0] = np.nan
#%%   
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(7.48, 3))
hg, = [ax1.plot(cp_HG['xf'][:, :12], cp_HG['yf'][:, :12], 's', 
         c='C0', mfc='None', markevery=8)]
disp, = [ax1.plot(cp_HG['xf'][:, 12:], cp_HG['yf'][:, 12:], 'o', 
         mec='tab:olive', mfc='none', markevery=8)]
mymodel, = [ax1.plot(f/1e6, cp, c='C1', lw=2)]

ax2.plot(att_HG['xf'][:, :12], att_HG['yf'][:, :12], 's', 
         c='C0', mfc='None', markevery=8)
ax2.plot(att_HG['xf'][:, 12:], att_HG['yf'][:, 12:], 'o', 
         mec='tab:olive', mfc='none', markevery=8)

ax2.plot(f/1e6, att, c='C1', lw=2)
ax1.set_xlim([0,3])
ax1.set_ylim([0, 10])
ax2.set_ylim([0, 2000])
ax2.set_xlabel('frequency in MHz')
ax1.set_xlabel('frequency in MHz')
leg = ax1.legend([hg[-1], disp[-1], mymodel[-1]], ['from Gravenkamp et al. [31]', 
                  'from Disperse [15]', 'present model'], loc=4, fontsize='x-small', 
                    frameon=True)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(0.8)

ax1.set_ylabel('phase velocity in km/s')
ax2.set_ylabel('attenuation in dB/m')
plt.tight_layout()
#plt.savefig('output_figures/tit_in_oil.pdf', dpi=600)
#%%
np.savetxt('Fig4_phase_velocity.txt', np.column_stack((f/1e6, cp)).real, fmt='%.4e', 
           delimiter=',', header='https://doi.org/10.1016/j.compstruc.2017.10.004 Fig.4\nFirst column - frequency in MHz, subsequent columns - phase velocities in km/s')
np.savetxt('Fig4_attenuation.txt', np.column_stack((f/1e6, att)).real, fmt='%.4e', 
           delimiter=',', header='https://doi.org/10.1016/j.compstruc.2017.10.004 Fig.4\nFirst column - frequency in MHz, subsequent columns - attenuations in dB/m')
