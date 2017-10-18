#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example illustrating how to use axisafe.

This file sets up simulation of elastic waves in a water-filled cast iron pipe
buried in soil.

@author: Michal K Kalkowski, kalkowski.m@gmail.com
Copyright (c) 2017 Michal Kalkowski (MIT license)
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('mkk_style.mplstyle')

# import axisafe
from context import axisafe
#%%
# define material properties
# use a miscallaneous function to calculate Lame coefficients from Young's modulus
# and Poisson's ration
lame_1, lame_2 = axisafe.misc.young2lame(156e9, 0.2)
# define pipe material [lame_1, lame_2, density, loss factor]
cast_iron = [lame_1, lame_2, 7800, 0.0]
# define soil material 
lame_1i, lame_2i = axisafe.misc.cLcS2lame(200, 100, 1500)
soil = [lame_1i, lame_2i, 1500, 0.]
# define fluid material [bulk modulus, density, loss factor]
water = [2.25e9, 1000, 0]

# define frequency vector
f = np.linspace(1e3, 20e3, 150, True)

# inner radius of the pipe: 0.1 m, wall thickness: 0.01 m
inner_r = 0.1
wall_thk = 0.01

# compute suggested PML parameters
# parameters: (start of PML, no of shortened wavelengths accross, waveguide material,
# soil material, f_start, f_end, allowed bulk attenuation)
h_long, h_short, PML_order, R_s, R_l, R_sl = \
        axisafe.mesh.suggest_PML_parameters(inner_r + wall_thk, 3, cast_iron, soil,
                                            f[0], f[-1], att=1)

# define element sets
# each sublist corresponds to one layer; for each layer one needs to define:
# [material label, starting coordinate, thickness, respective material list, element type]
# in case of a PML layer, one needs also to provide the order of the PML element
# computed above; it is also advised to use the thickness of the PML as suggested by 
# function called above
sets = [['water', 0.0, inner_r, water, 'ALAX6'], 
        ['cast_iron', inner_r, wall_thk, cast_iron, 'SLAX6'], 
        ['soil', inner_r + wall_thk, np.round(h_short, 4), soil, 'SLAX6_PML', PML_order]]

#%%
# initiate a mesh object
mesh0 = axisafe.mesh.Mesh()
# create the mesh
# we assign sets, the highest frequency of interest, identify the PML layer and
# provide PML parameters (only if a PML exists, of course)
mesh0.create(sets, f[-1], PML='soil', 
             PML_props=[inner_r + wall_thk, np.round(h_short, 4), 6 + 7j])
# assemble global SAFE matrices for a given circumferential order (here n=0)
mesh0.assemble_matrices(n=0)

# initiate a wave element based on the previously created mesh
pipe0 = axisafe.solver.WaveElementAxisym(mesh0)
# solve for a given frequency vector
pipe0.solve(f)
# calculate the ratio of kinetic energies in the core waveguide and in the PML
pipe0.energy_ratio()
# extract propagating waves only, given maximum imaginary k and energy ratio
# the output of this procedure is written into pipe0.k_ready array
pipe0.k_propagating(15, 0.935)
k_0 = np.copy(pipe0.k_ready)

#%% Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(7.48, 3))
safe, = [ax1.plot(f/1e3, k_0.real, '.-')]
ax2.plot(f/1e3, -20*np.log10(np.e)*k_0.imag, '.-')
ax1.legend([safe[0]], ['axisafe model'], loc=2)
ax1.set_xlim([0,20])
ax2.set_ylim([0, 40])
ax2.set_xlabel('frequency in kHz')
ax1.set_xlabel('frequency in kHz')

ax1.set_ylabel(' real k in rad/m')
ax2.set_ylabel(' attenuation in dB/m')
ax1.grid(lw=0.)
ax2.grid(lw=0.)
plt.tight_layout()
#plt.savefig('cast_iron_water_pipe_in_soil.pdf', transparent=True, dpi=600)