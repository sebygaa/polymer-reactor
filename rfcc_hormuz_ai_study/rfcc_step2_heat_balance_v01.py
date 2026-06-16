#!/usr/bin/env python3
"""
rfcc_step2_heat_balance_v01.py
RFCC crisis-response study -- Step 2: integrated riser + regenerator.

The riser and regenerator are closed through the unit heat balance
(coke yield -> regen T -> required C/O -> conversion/coke), which is what
makes heavy-feed operation hard.  Studies:
  (a) base-case summary table
  (b) ROT setpoint sweep         (severity lever)
  (c) feed preheat sweep         (heat-balance lever: preheat down -> C/O up)
  (d) feed CCR sweep             (the heavy-feed squeeze)
  (e) operating-window contours: margin over (ROT, preheat) for normal and
      crisis-heavy feed, with the T_rgn <= 730 degC wall

Outputs: rfcc_step2_heat_balance_v01.png
         rfcc_step2_operating_window_v01.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from rfcc_model import (solve_unit, economics, T_RGN_MAX_C,
                        ROT_LO, ROT_HI, PRE_LO, PRE_HI)

t_start = time.time()

# ---------------------------------------------------------------- (a)
base = solve_unit(520.0, 220.0, 0.5, CCR=4.0)
eco = economics(base)
print('=' * 64)
print('RFCC base case (ROT 520 degC, preheat 220 degC, cooler 50 %,')
print('                CCR 4 wt%, design rate)')
print('=' * 64)
rows = [('conversion', base['conv'], '-'),
        ('gasoline yield', base['y_gaso'], 'wt frac'),
        ('LCO yield', base['y_lco'], 'wt frac'),
        ('gas yield (LPG+dry)', base['y_gas'], 'wt frac'),
        ('coke yield', base['y_coke'], 'wt frac'),
        ('slurry (unconv.)', base['y_hs'], 'wt frac'),
        ('C/O ratio', base['CO'], '-'),
        ('regen temperature', base['T_rgn_C'], 'degC'),
        ('riser mix temperature', base['T_mix_C'], 'degC'),
        ('air rate', base['F_air'], 'kg/s'),
        ('margin', eco['margin_per_t'], '$/t'),
        ('daily margin', eco['margin_kUSD_day'], 'k$/day')]
for name, val, unit in rows:
    print(f'  {name:24s} {val:10.3f}  {unit}')

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

def sweep(ax_, xs, results, xlabel, title):
    eco_ = [economics(r) for r in results]
    ax_.plot(xs, [r['conv'] for r in results], 'k-', lw=2.5, label='conversion')
    ax_.plot(xs, [r['y_gaso'] for r in results], color='#d62728', lw=2,
             label='gasoline')
    ax_.plot(xs, [r['y_coke'] for r in results], color='#8c564b', lw=2,
             label='coke')
    ax_.set_xlabel(xlabel); ax_.set_ylabel('mass fraction / conversion [-]')
    ax_.grid(alpha=0.3); ax_.legend(loc='center left', fontsize=8)
    ax2 = ax_.twinx()
    ax2.plot(xs, [r['T_rgn_C'] for r in results], 'b--', lw=1.8,
             label='T_rgn')
    ax2.axhline(T_RGN_MAX_C, color='b', ls=':', lw=1)
    ax2.plot(xs, [e['margin_per_t'] for e in eco_], 'g-.', lw=1.8,
             label='margin')
    ax2.set_ylabel('T_rgn [degC]  /  margin [$/t]', color='b')
    ax2.legend(loc='center right', fontsize=8)
    ax_.set_title(title)

# ---------------------------------------------------------------- (b)
rots = np.linspace(ROT_LO, ROT_HI, 16)
res_b = [solve_unit(r, 220.0, 0.5, 4.0) for r in rots]
sweep(ax[0, 0], rots, res_b, 'ROT setpoint [degC]',
      '(b) Severity sweep: riser outlet temperature')

# ---------------------------------------------------------------- (c)
pres = np.linspace(PRE_LO, PRE_HI, 16)
res_c = [solve_unit(520.0, p, 0.5, 4.0) for p in pres]
sweep(ax[0, 1], pres, res_c, 'feed preheat [degC]',
      '(c) Heat-balance sweep: feed preheat')
ax2 = ax[0, 1].twinx()  # annotate C/O response on a light axis
co_ = [r['CO'] for r in res_c]
ax[0, 1].annotate(f"C/O: {co_[0]:.1f} -> {co_[-1]:.1f}",
                  xy=(0.05, 0.92), xycoords='axes fraction', fontsize=9)
ax2.set_yticks([])

# ---------------------------------------------------------------- (d)
ccrs = np.linspace(1.0, 9.0, 17)
res_d = [solve_unit(520.0, 220.0, 0.5, c) for c in ccrs]
sweep(ax[1, 0], ccrs, res_d, 'feed CCR [wt%]',
      '(d) Heavy-feed squeeze: CCR at design operation')

# ---------------------------------------------------------------- (e0)
qcs = np.linspace(0.0, 1.0, 11)
res_e = [solve_unit(520.0, 220.0, q, 6.0) for q in qcs]
sweep(ax[1, 1], qcs, res_e, 'catalyst cooler duty fraction [-]',
      '(e) Cat-cooler sweep at heavy feed (CCR 6 wt%)')

fig.suptitle('RFCC Step 2 -- integrated riser/regenerator heat balance',
             fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig('rfcc_step2_heat_balance_v01.png', dpi=130, bbox_inches='tight')
print('saved rfcc_step2_heat_balance_v01.png')

# ---------------------------------------------------------------- (e)
# operating-window maps: margin(ROT, preheat) with the T_rgn wall
NG = 15
rot_g = np.linspace(ROT_LO, ROT_HI, NG)
pre_g = np.linspace(PRE_LO, PRE_HI, NG)
fig2, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for axx, ccr, qc, lbl in [(axes[0], 4.0, 0.5, 'normal feed CCR 4 wt%'),
                          (axes[1], 6.8, 1.0, 'crisis heavy feed CCR 6.8 wt%'
                                              ' (cooler max)')]:
    M = np.zeros((NG, NG)); Trg = np.zeros((NG, NG))
    for i, p in enumerate(pre_g):
        for j, r in enumerate(rot_g):
            res = solve_unit(r, p, qc, ccr)
            M[i, j] = economics(res)['margin_per_t']
            Trg[i, j] = res['T_rgn_C']
    cs = axx.contourf(rot_g, pre_g, M, levels=14, cmap='RdYlGn')
    plt.colorbar(cs, ax=axx, label='margin [$/t feed]')
    axx.contourf(rot_g, pre_g, Trg, levels=[T_RGN_MAX_C, 2000],
                 colors='none', hatches=['////'])
    axx.contour(rot_g, pre_g, Trg, levels=[T_RGN_MAX_C], colors='k',
                linewidths=2)
    ij = np.unravel_index(np.argmax(np.where(Trg <= T_RGN_MAX_C, M, -1e9)),
                          M.shape)
    axx.plot(rot_g[ij[1]], pre_g[ij[0]], 'b*', ms=16,
             label=f'best feasible: {M[ij]:.0f} $/t')
    axx.plot(520, 220, 'ks', ms=9, label='design point')
    axx.set_xlabel('ROT setpoint [degC]'); axx.set_ylabel('feed preheat [degC]')
    axx.set_title(f'(e) Operating window -- {lbl}\n'
                  '(hatched: T_rgn > 730 degC infeasible)')
    axx.legend(loc='lower left', fontsize=8)
fig2.tight_layout()
fig2.savefig('rfcc_step2_operating_window_v01.png', dpi=130,
             bbox_inches='tight')
print('saved rfcc_step2_operating_window_v01.png')
print(f'total runtime {time.time()-t_start:.1f} s')
