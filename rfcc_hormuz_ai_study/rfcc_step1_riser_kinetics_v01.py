#!/usr/bin/env python3
"""
rfcc_step1_riser_kinetics_v01.py
RFCC crisis-response study -- Step 1: riser 5-lump kinetics.

Riser-only studies (regenerator temperature fixed, no unit heat balance):
  (a) lump profiles and temperature along the riser at base conditions
  (b) outlet yields vs catalyst-to-oil ratio (C/O)
  (c) gasoline overcracking: gasoline yield vs residence time at several
      mix temperatures (the classical FCC yield optimum)
  (d) feed-quality (CCR) effect on conversion and coke at fixed C/O

Output: rfcc_step1_riser_kinetics_v01.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rfcc_model import riser, mix_temperature, T_RES

# base riser-only conditions (regen T and preheat fixed by hand here;
# the full unit heat balance is Step 2)
T_RGN_K = 668.0 + 273.15
T_PRE_K = 220.0 + 273.15
CO_BASE = 9.8
CCR_BASE = 4.0

LUMP_NAMES = ['HS (slurry)', 'LCO', 'Gasoline', 'Gas', 'Coke']
LUMP_COLORS = ['#555555', '#1f77b4', '#d62728', '#2ca02c', '#8c564b']

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# ---------------------------------------------------------------- (a)
T_mix = mix_temperature(CO_BASE, T_RGN_K, T_PRE_K)
ts, Y = riser(CO_BASE, T_mix, CCR_BASE, profile=True)
a = ax[0, 0]
for i in range(5):
    a.plot(ts, Y[:, i], color=LUMP_COLORS[i], label=LUMP_NAMES[i], lw=2)
a.set_xlabel('riser residence time [s]')
a.set_ylabel('mass fraction [-]')
a2 = a.twinx()
a2.plot(ts, Y[:, 5] - 273.15, 'k--', lw=1.5, label='T')
a2.set_ylabel('T [degC]')
a.set_title(f'(a) Riser profiles  (C/O={CO_BASE}, CCR={CCR_BASE} wt%, '
            f'T_mix={T_mix-273.15:.0f} degC)')
a.legend(loc='center right', fontsize=8)
a.grid(alpha=0.3)
print(f"(a) outlet: conv={1-Y[-1,0]-Y[-1,1]:.3f}  gaso={Y[-1,2]:.3f}  "
      f"coke={Y[-1,4]:.3f}  T_out={Y[-1,5]-273.15:.1f} degC")

# ---------------------------------------------------------------- (b)
a = ax[0, 1]
COs = np.linspace(3, 14, 23)
outs = np.array([riser(co, mix_temperature(co, T_RGN_K, T_PRE_K), CCR_BASE)
                 for co in COs])
for i in range(5):
    a.plot(COs, outs[:, i], color=LUMP_COLORS[i], label=LUMP_NAMES[i], lw=2)
a.plot(COs, 1 - outs[:, 0] - outs[:, 1], 'k-', lw=2.5, label='conversion')
a.set_xlabel('catalyst-to-oil ratio C/O [-]')
a.set_ylabel('outlet mass fraction [-]')
a.set_title(f'(b) Yields vs C/O  (T_rgn={T_RGN_K-273.15:.0f} degC fixed)')
a.legend(fontsize=8)
a.grid(alpha=0.3)

# ---------------------------------------------------------------- (c)
a = ax[1, 0]
times = np.linspace(0.5, 10.0, 40)
for T_mix_C, c in [(530, '#9ecae1'), (560, '#4292c6'), (590, '#08519c'),
                   (620, '#08306b')]:
    g = [riser(CO_BASE, T_mix_C + 273.15, CCR_BASE, t_res=t,
               n_step=max(30, int(30 * t)))[2]
         for t in times]
    a.plot(times, g, color=c, lw=2, label=f'T_mix = {T_mix_C} degC')
    iopt = int(np.argmax(g))
    a.plot(times[iopt], g[iopt], 'o', color=c, ms=6)
a.axvline(T_RES, color='gray', ls=':', lw=1)
a.text(T_RES + 0.1, 0.1, 'design t_res', fontsize=8, color='gray')
a.set_xlabel('residence time [s]')
a.set_ylabel('gasoline yield [-]')
a.set_title('(c) Gasoline overcracking optimum (markers = max)')
a.legend(fontsize=8)
a.grid(alpha=0.3)

# ---------------------------------------------------------------- (d)
a = ax[1, 1]
CCRs = np.linspace(0.5, 9.0, 30)
outs = np.array([riser(CO_BASE, mix_temperature(CO_BASE, T_RGN_K, T_PRE_K),
                       ccr) for ccr in CCRs])
a.plot(CCRs, 1 - outs[:, 0] - outs[:, 1], 'k-', lw=2.5, label='conversion')
a.plot(CCRs, outs[:, 2], color=LUMP_COLORS[2], lw=2, label='gasoline')
a.plot(CCRs, outs[:, 4], color=LUMP_COLORS[4], lw=2, label='coke')
a.set_xlabel('feed CCR [wt%]')
a.set_ylabel('outlet mass fraction [-]')
a.set_title(f'(d) Feed-quality effect at fixed C/O={CO_BASE}\n'
            '(deactivation + additive coke)')
a.legend(fontsize=8)
a.grid(alpha=0.3)

fig.suptitle('RFCC Step 1 -- riser 5-lump kinetics (riser-only)',
             fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig('rfcc_step1_riser_kinetics_v01.png', dpi=130,
            bbox_inches='tight')
print('saved rfcc_step1_riser_kinetics_v01.png')
