#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:27:20 2017

Validation of SAFE routines with spectral elements
Steel rod in concrete, following Nguyen et el.

@author: michal
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
plt.style.use('jsv.mplstyle')

from context import axisafe
#%%
lame_1, lame_2 = axisafe.misc.cLcS2lame(5960, 3260, 7932)
steel = [lame_1, lame_2, 7932, 0.000]
lame_1i, lame_2i = axisafe.misc.cLcS2lame(4222.1, 2637.5, 2300)
concrete = [lame_1i, lame_2i, 2300, 0.000]

h_long, h_short, PML_order, R_s, R_l, R_sl = \
        axisafe.mesh.suggest_PML_parameters(0.01, 6, steel, 
                                  concrete, 100, 2e5, 
                                  att=10, alpha=6, beta=7)

sets = [['steel', 0, 0.01, steel, 'SLAX6'], 
        ['concrete', 0.01, np.round(h_short, 4), concrete, 'SLAX6_PML', PML_order]]
        

f = np.linspace(1, 2e5, 100, True)
#%%
#%%
mesh0 = axisafe.mesh.Mesh()
mesh0.create(sets, f[-1], PML='concrete', 
             PML_props=[0.01, np.round(h_short, 4), 6 + 7j])
mesh0.assemble_matrices(n=0)

rod0 = axisafe.solver.WaveElementAxisym(mesh0)
rod0.solve(f)
rod0.energy_ratio()
rod0.k_propagating(200, 0.9)
k_0 = np.copy(rod0.k_ready)
#%%
mesh1 = axisafe.mesh.Mesh()
mesh1.create(sets, f[-1], PML='concrete', 
             PML_props=[0.01, np.round(h_short, 4), 6 + 7j])
mesh1.assemble_matrices(n=1)

rod1 = axisafe.solver.WaveElementAxisym(mesh1)
rod1.solve(f)
rod1.energy_ratio()
rod1.k_propagating(200, 0.9)
k_1 = np.copy(rod1.k_ready)
#%%
mesh2 = axisafe.mesh.Mesh()
mesh2.create(sets, f[-1], PML='concrete', 
             PML_props=[0.01, np.round(h_short, 4), 6 + 7j])
mesh2.assemble_matrices(n=2)

rod2 = axisafe.solver.WaveElementAxisym(mesh2)
rod2.solve(f)
rod2.energy_ratio()
rod2.k_propagating(200, 0.9)
k_2 = np.copy(rod2.k_ready)
#%%
mesh3 = axisafe.mesh.Mesh()
mesh3.create(sets, f[-1], PML='concrete', 
             PML_props=[0.01, np.round(h_short, 4), 6 + 7j])
mesh3.assemble_matrices(n=3)

rod3 = axisafe.solver.WaveElementAxisym(mesh3)
rod3.solve(f)
rod3.energy_ratio()
rod3.k_propagating(200, 0.9)
k_3 = np.copy(rod3.k_ready)
#%%
cp_duan = np.genfromtxt('reference/duan/duan.csv', delimiter=',', skip_header=2)
att_duan = np.genfromtxt('reference/duan/attenuation.csv', delimiter=',', skip_header=2)

cp_nguyen = loadmat('reference/nguyen/cp.mat')
att_nguyen = loadmat('reference/nguyen/att.mat')
#%%
# Plot results
for results_arrray in [k_0, k_1, k_2, k_3]:
    results_arrray[results_arrray.real == 0] = np.nan + 1j*np.nan
    results_arrray[results_arrray.imag == 0] = np.nan + 1j*np.nan
cp_nguyen['final2'][cp_nguyen['final2'] == 0] = np.nan
att_nguyen['final2'][att_nguyen['final2'] == 0] = np.nan
att_duan[att_duan == 0] =np.nan

plt.rcParams.update({'figure.figsize': [7.48, 3]})
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
lines0, = [ax1.plot(cp_duan[:, 0::2], cp_duan[:, 1::2]/1e3, 's', mec='tab:olive', mfc='None', 
         markevery=6)]
lines1, = [ax1.plot(cp_nguyen['final2'][:, 0], cp_nguyen['final2'][:, 1:]/1e3, 'o', 
         mec='tab:blue', mfc='None')]
lines2, = [ax1.plot(f/1e3, 2e-3*np.pi*f.reshape(-1, 1)/k_0.real, '-', c='tab:orange', lw=2)]

ax1.plot(f/1e3, 2e-3*np.pi*f.reshape(-1, 1)/k_1.real, '-', c='tab:orange', lw=2)
ax1.plot(f/1e3, 2e-3*np.pi*f.reshape(-1, 1)/k_2.real, '-', c='tab:orange', lw=2)
ax1.plot(f/1e3, 2e-3*np.pi*f.reshape(-1, 1)/k_3.real, '-', c='tab:orange', lw=2)

ax2.plot(att_duan[:, 0::2], att_duan[:, 1::2], 's', mec='tab:olive', mfc='None', 
         markevery=6)
ax2.plot(att_nguyen['final2'][:, 0], att_nguyen['final2'][:, 1:], 'o', 
         mec='tab:blue', mfc='None')
ax2.plot(f/1e3, -20*np.log10(np.e)*k_0.imag, '-', c='tab:orange', lw=2)
ax2.plot(f/1e3, -20*np.log10(np.e)*k_1.imag, '-', c='tab:orange', lw=2)
ax2.plot(f/1e3, -20*np.log10(np.e)*k_2.imag, '-', c='tab:orange', lw=2)
ax2.plot(f/1e3, -20*np.log10(np.e)*k_3.imag, '-', c='tab:orange', lw=2)

ax1.plot(f/1e3, len(f)*[4.2221], '--', lw=0.5, c='black')
ax1.plot(f/1e3, len(f)*[2.6375], '--', lw=0.5, c='black')
ax1.set_ylim([0, 10])
ax2.set_ylim([0, 920])
ax2.set_xlabel('frequency in kHz')
ax1.set_xlabel('frequency in kHz')
ax1.set_ylabel(' phase velocity in km/s')
# Annotations
ax1.annotate("T(0, 1)",
            xy=(9.52, 1.85), xycoords='data',
            xytext=(24, -15), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax1.annotate("L(0, 1)",
            xy=(67.17, 5.25), xycoords='data',
            xytext=(-8, 15), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax1.annotate("F(0, 1)",
            xy=(87.6, 2.74), xycoords='data',
            xytext=(12, -15), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax1.annotate("F(2, 1)",
            xy=(62.1, 3.46), xycoords='data',
            xytext=(-12, 0), textcoords='offset points', ha='right',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax1.annotate("L(0, 2)",
            xy=(141.6, 5.42), xycoords='data',
            xytext=(-5, 8), textcoords='offset points', ha='right', va='bottom',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax1.annotate("F(1, 2)",
            xy=(119.7, 8.95), xycoords='data',
            xytext=(-40, -25), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax1.annotate("F(1, 3)",
            xy=(179., 7.25), xycoords='data',
            xytext=(-18, 15), textcoords='offset points', ha='right', va='bottom',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax1.annotate("F(2, 2)",
            xy=(189.05, 9.13), xycoords='data',
            xytext=(12, 6), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax1.annotate("F(3, 1)",
            xy=(186.85, 5.55), xycoords='data',
            xytext=(-15, -7), textcoords='offset points', ha='right', va='top',
            arrowprops=dict(arrowstyle="->", shrinkA=-0.3,
                            connectionstyle="arc3, rad=0"))
ax1.annotate("L(0, 1)",
            xy=(163, 3.87), xycoords='data',
            xytext=(15,2), textcoords='offset points', va='center', ha='left',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))

ax2.annotate("T(0, 1)",
            xy=(13.5, 878.3), xycoords='data',
            xytext=(0, -25), textcoords='offset points', va='top', ha='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax2.annotate("L(0, 1)",
            xy=(18.75, 163.52), xycoords='data',
            xytext=(12, 0), textcoords='offset points', ha='left',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax2.annotate("F(0, 1)",
            xy=(55.1, 50.35), xycoords='data',
            xytext=(-18, 0), textcoords='offset points', ha='right', 
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax2.annotate("F(2, 1)",
            xy=(114.35, 866), xycoords='data',
            xytext=(-12, 0), textcoords='offset points', ha='right',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax2.annotate("L(0, 2)",
            xy=(161.74, 866.6), xycoords='data',
            xytext=(14, 3), textcoords='offset points', ha='left',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax2.annotate("F(1, 2)",
            xy=(63.28, 717.44), xycoords='data',
            xytext=(12, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax2.annotate("F(1, 3)",
            xy=(176.3, 420.16), xycoords='data',
            xytext=(-14, 0), textcoords='offset points', ha='right',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax2.annotate("F(2, 2)",
            xy=(156.64, 662), xycoords='data',
            xytext=(-35, 0), textcoords='offset points', ha='right',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax2.annotate("F(3, 1)",
            xy=(196.2, 735.4), xycoords='data',
            xytext=(0, -25), textcoords='offset points', ha='center', va='top',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax2.annotate("L(0, 1)",
            xy=(180.49, 124.21), xycoords='data',
            xytext=(-15, 0), textcoords='offset points', ha='right',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3, rad=0"))
ax1.annotate(r"$c_L$ concrete", xy=(24, 4.22), xycoords='data', xytext=(0, 0), 
             textcoords='offset points', va='bottom', fontsize='x-small')
ax1.annotate(r"$c_S$ concrete", xy=(17.9, 2.64), xycoords='data', xytext=(0, 0), 
             textcoords='offset points', va='bottom', fontsize='x-small')
ax2.set_ylabel(' attenuation in dB/m')
leg = ax1.legend([lines0[-1], lines1[-1], lines2[-1]], 
            ['from Duan et al. [28]', 'from Nguyen et al. [26]', 'present model'], 
            loc=2, fontsize='x-small', frameon=True)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(0.8)
#for ax in [ax1, ax2]:
#    ax.spines['right'].set_color('none')
#    ax.spines['top'].set_color('none')
#    ax.xaxis.set_ticks_position('bottom')
#    ax.yaxis.set_ticks_position('left')
#ax1.grid(lw=0.0)
#ax2.grid(lw=0.0)

plt.tight_layout()
#plt.savefig('output_figures/steel_concrete.pdf', transparent=True, dpi=600)
plt.show()
