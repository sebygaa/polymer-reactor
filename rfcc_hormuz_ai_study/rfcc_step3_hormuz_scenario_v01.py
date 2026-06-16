#!/usr/bin/env python3
"""
rfcc_step3_hormuz_scenario_v01.py
RFCC crisis-response study -- Step 3: Strait-of-Hormuz crisis scenario
and the baseline (design-book + operator emergency trims) response.

  (1) generate the 150-day crisis timeline (feed CCR / supply / prices)
  (2) run the baseline operating policy day by day on the physics model
  (3) quantify the crisis cost vs a no-crisis counterfactual

Outputs: rfcc_step3_scenario_v01.png
         rfcc_step3_baseline_response_v01.png
         rfcc_step3_baseline_v01.npz   (reused by Step 5)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from rfcc_model import (scenario_timeline, prices_of_day, baseline_day,
                        PRICES_BASE, T_RGN_MAX_C)

t_start = time.time()
scen = scenario_timeline(n_days=150, seed=42)
days = scen['days']
n_days = len(days)

PHASE_NAMES = ['normal', 'blockade\nshock', 'adaptation\n(light sweet)',
               'partial\nreopen', 'normali-\nsation']
PHASE_EDGES = [0, 15, 35, 70, 110, 150]
PHASE_COLORS = ['#ffffff', '#ffcccc', '#fff2cc', '#e2efda', '#ffffff']


def shade_phases(ax):
    for k in range(5):
        ax.axvspan(PHASE_EDGES[k], PHASE_EDGES[k + 1],
                   color=PHASE_COLORS[k], alpha=0.5, zorder=0)


# ---------------------------------------------------------------- (1)
fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
a = ax[0]
shade_phases(a)
a.plot(days, scen['CCR'], 'k-', lw=2)
a.set_ylabel('feed CCR [wt%]')
a.set_title('Hormuz-strait crisis scenario (150 days)')
for k in range(5):
    a.text(0.5 * (PHASE_EDGES[k] + PHASE_EDGES[k + 1]),
           a.get_ylim()[1] * 0.93, PHASE_NAMES[k], ha='center', fontsize=8)
a.grid(alpha=0.3)

a = ax[1]
shade_phases(a)
a.plot(days, scen['supply'] * 100, 'b-', lw=2)
a.set_ylabel('feed availability [% design]')
a.grid(alpha=0.3)

a = ax[2]
shade_phases(a)
a.plot(days, scen['p_feed'], 'k-', lw=2, label='feed cost')
a.plot(days, scen['p_gaso'], color='#d62728', lw=2, label='gasoline')
a.plot(days, scen['p_lco'], color='#1f77b4', lw=2, label='LCO (diesel)')
a.set_ylabel('price [$/t]')
a.set_xlabel('day')
a.legend(fontsize=9)
a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('rfcc_step3_scenario_v01.png', dpi=130, bbox_inches='tight')
print('saved rfcc_step3_scenario_v01.png')

# ---------------------------------------------------------------- (2)
keys_ops = ['ROT_C', 'T_pre_C', 'Qcool_frac', 'feed_frac']
keys_res = ['conv', 'y_gaso', 'y_coke', 'CO', 'T_rgn_C', 'viol_Trgn']
log = {k: np.zeros(n_days) for k in keys_ops + keys_res
       + ['margin_kUSD_day', 'margin_normalfix_kUSD_day']}

for d in days:
    prices = prices_of_day(scen, d)
    ops, res, eco = baseline_day(scen['CCR'][d], scen['supply'][d], prices)
    for k in keys_ops:
        log[k][d] = ops[k]
    for k in keys_res:
        log[k][d] = res[k]
    log['margin_kUSD_day'][d] = eco['margin_kUSD_day']
    # no-crisis counterfactual: normal feed, full supply, base prices
    if d == 0:
        _, _, eco0 = baseline_day(4.0, 1.0, PRICES_BASE)
    log['margin_normalfix_kUSD_day'][d] = eco0['margin_kUSD_day']
print(f'baseline timeline solved in {time.time()-t_start:.1f} s')

cum = np.cumsum(log['margin_kUSD_day']) / 1e3          # M$
cum0 = np.cumsum(log['margin_normalfix_kUSD_day']) / 1e3
crisis_cost = cum0[-1] - cum[-1]

# ---------------------------------------------------------------- (3)
fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
a = ax[0]
shade_phases(a)
a.plot(days, log['margin_kUSD_day'], 'k-', lw=2, label='baseline (crisis)')
a.plot(days, log['margin_normalfix_kUSD_day'], 'g--', lw=1.5,
       label='no-crisis counterfactual')
a.set_ylabel('margin [k$/day]')
a.set_title('Baseline response: design-book operation + operator trims\n'
            f'crisis cost vs counterfactual = {crisis_cost:.1f} M$ / 150 d')
a.legend(fontsize=9)
a.grid(alpha=0.3)

a = ax[1]
shade_phases(a)
a.plot(days, log['ROT_C'], color='#d62728', lw=2, label='ROT [degC]')
a.set_ylabel('ROT [degC]', color='#d62728')
a2 = a.twinx()
a2.plot(days, log['feed_frac'] * 100, 'b-', lw=2, label='feed rate')
a2.plot(days, scen['supply'] * 100, 'b:', lw=1.2, label='available supply')
a2.plot(days, log['Qcool_frac'] * 100, color='#7f7f7f', lw=1.5,
        label='cat cooler duty')
a2.set_ylabel('feed rate / supply / cooler [%]', color='b')
a2.legend(fontsize=8, loc='lower right')
a.set_title('Operating conditions (reactive trims only)')
a.grid(alpha=0.3)

a = ax[2]
shade_phases(a)
a.plot(days, log['T_rgn_C'], 'b-', lw=2)
a.axhline(T_RGN_MAX_C, color='r', ls='--', lw=1.5, label='regen limit')
a.set_ylabel('T_rgn [degC]')
a.legend(fontsize=9)
a.set_title('Regenerator temperature (heat-balance stress)')
a.grid(alpha=0.3)

a = ax[3]
shade_phases(a)
a.plot(days, log['conv'], 'k-', lw=2, label='conversion')
a.plot(days, log['y_gaso'], color='#d62728', lw=2, label='gasoline yield')
a.plot(days, log['y_coke'] * 4, color='#8c564b', lw=2, label='coke yield x4')
a.set_ylabel('fraction [-]')
a.set_xlabel('day')
a.legend(fontsize=9)
a.set_title('Conversion and key yields')
a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('rfcc_step3_baseline_response_v01.png', dpi=130,
            bbox_inches='tight')
print('saved rfcc_step3_baseline_response_v01.png')

np.savez('rfcc_step3_baseline_v01.npz',
         **{k: v for k, v in log.items()},
         **{('scen_' + k): v for k, v in scen.items()})
print('saved rfcc_step3_baseline_v01.npz')

print()
print(f"cumulative baseline margin (crisis)       : {cum[-1]:8.1f} M$")
print(f"cumulative counterfactual margin (normal) : {cum0[-1]:8.1f} M$")
print(f"crisis cost under baseline operation      : {crisis_cost:8.1f} M$")
for k in range(5):
    sl = slice(PHASE_EDGES[k], PHASE_EDGES[k + 1])
    print(f"  phase {k} ({PHASE_NAMES[k].replace(chr(10),' '):24s}): "
          f"mean margin {log['margin_kUSD_day'][sl].mean():7.0f} k$/day")
print(f'total runtime {time.time()-t_start:.1f} s')
