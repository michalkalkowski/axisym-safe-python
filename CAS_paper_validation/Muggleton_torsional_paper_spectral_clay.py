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
plt.style.use('/home/michal/Dropbox/plot_templates/default.mplstyle')

from context import axisafe
#%%
lame_1, lame_2 = axisafe.misc.G_nu2lame(0.6e9, 0.4)
cast_iron = [lame_1, lame_2, 2000, 0.065]
lame_1i, lame_2i = axisafe.misc.G_nu2lame(0.18e9, 0.3)
soil = [lame_1i, lame_2i, 2000, 0.1]
properties = dict([('cast_iron', cast_iron), ('soil', soil)])

h_long, h_short, PML_order, R_s, R_l, R_sl= \
        axisafe.mesh.suggest_PML_parameters(0.105, 8, properties['cast_iron'], 
                                  properties['soil'], 1, 5e3, att=1)

sets = [['cast_iron', 0.095, 0.01, cast_iron, 'SLAX6'], 
        ['soil', 0.105, h_long/16, soil, 'SLAX6_PML', PML_order, 1]] 
        
#f = np.logspace(-1, 3.7, 50, True)
f = np.linspace(1, 5e3, 300, True)
#%%
mesh0 = axisafe.mesh.Mesh()
mesh0.create(sets, f[-1], 0, 'soil', PML_props=[0.105, h_long/16, 6 + 7j])
mesh0.assemble_matrices(n=0)

pipe0 = axisafe.solver.WaveElementAxisym(mesh0)
pipe0.solve(f, no_of_waves='full',
                    central_wavespeed=((cast_iron[0] + 2*cast_iron[1])/cast_iron[2])**0.5)
pipe0.energy_ratio()
pipe0.k_propagating(20, 0.95)

k_0 = pipe0.k[:, pipe0.propagating_indices]

#%% From Jen Muggleton's torsional waves paper in JSV 2016
#Jen_r = np.genfromtxt('/home/michal/mCloud/ATU/torsional_waves_Jen/results/Jens/Rek_Jen.txt')
#Jen_i = np.genfromtxt('/home/michal/mCloud/ATU/torsional_waves_Jen/results/Jens/Imk_Jen.txt')
#Jens = np.column_stack((Jen_r[:, 0], Jen_r[:, 3],  Jen_i[:, 1]))
kfa = 0.1*(2*np.pi*f)/1500
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=((7.48, 3)))
safe, = [ax1.plot(kfa, 0.1*k_0.real)]
ax2.plot(kfa, -0.1*20*np.log10(np.e)*k_0.imag)
#jen, = ax1.plot(Jens[:, 0]*1500/(0.1*2*np.pi), Jens[:, 1], lw=3, color='red')
#ax2.plot(Jens[:, 0]*1500/(0.1*2*np.pi), Jens[:, 2], lw=3, color='red')
#ax1.legend([safe[0], safe[1], jen], ['SAFE axial', 'SAFE torsional', 'Jen\'s torsional'], loc=2)
ax1.set_xlim([0, 2])
ax1.set_ylim([0, 7])
ax2.set_ylim([0, 30])
ax2.set_xlabel('frequency in Hz')
ax1.set_ylabel(' real kr')
ax2.set_ylabel(' attenuation in dB/r')
for ax in [ax1, ax2]:
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
plt.tight_layout()
plt.show()
#%% 
