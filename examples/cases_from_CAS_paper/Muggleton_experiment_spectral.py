#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of SAFE routines with spectral elements
:Waves in a water-filled MDPE pipe buried in soil:
    
Computes a case found in:
- Muggleton, J.M., Brennan, M.J., Linford, P.W., 2004. Axisymmetric wave propagation
in fluid-filled pipes: wavenumber measurements in in vacuo and buried pipes. 
Journal of Sound and Vibration 270, 171–190. doi:10.1016/S0022-460X(03)00489-9
- Gao, Y., Sui, F., Muggleton, J.M., Yang, J., 2016. Simplified dispersion 
relationships for fluid-dominated axisymmetric wave motion in buried fluid-filled 
pipes. Journal of Sound and Vibration 375, 386–402. doi:10.1016/j.jsv.2016.04.012    
    
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
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(7.48, 3))
safe, = [ax1.plot(  f, k_0[:, 2].real, '.-')]
ax2.plot(f, -20*np.log10(np.e)*k_0[:, 2].imag, '.-')
ax1.legend([safe[0]], ['axisafe model'], loc=2)
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