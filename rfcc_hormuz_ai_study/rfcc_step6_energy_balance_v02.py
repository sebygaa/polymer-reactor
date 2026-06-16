#!/usr/bin/env python3
"""
rfcc_step6_energy_balance_v02.py
RFCC crisis-response study -- Step 6: reactor/regenerator split (v02)
and energy-balance verification.

  (a) v01 vs v02 base case + behaviour sweeps (CCR, ROT) -- confirms the
      restructured two-vessel model reproduces the v01 unit
  (b) energy-balance term breakdown for both vessels (base & crisis feed)
  (c) closure-residual scan over the whole operating x feed space.
      What the closure audit verifies: every enthalpy stream definition
      (mix-T formula, riser energy ODE, regenerator bed balance) is
      mutually consistent -- any bookkeeping drift between the solver and
      the audit shows up as a nonzero residual.  Bed-clamp (regen_cold)
      points carry their heat deficit as a flagged, quantified residual.
  (d) discretisation accuracy by step refinement: |T_out(n) - T_out(960)|
      and |q_rxn(n) - q_rxn(960)| converge at the expected RK4 order
      (the closure residual itself is an exact linear identity of the
      integrator, so accuracy must be verified independently like this)
  (e) operator-trim / baseline policy check on the v02 unit

Outputs: rfcc_step6_energy_balance_v02.png
         rfcc_step6_v01_v02_comparison.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import rfcc_model as m1
import rfcc_model_v02 as m2
from rfcc_model_v02 import RiserReactor, RFCCUnit

t_start = time.time()
rng = np.random.default_rng(21)

# ---------------------------------------------------------------- (a)
print('=' * 72)
print('(a) v01 (lumped heat balance) vs v02 (two-vessel) -- base case')
print('=' * 72)
r1 = m1.solve_unit(520.0, 220.0, 0.5, CCR=4.0)
r2 = m2.solve_unit(520.0, 220.0, 0.5, CCR=4.0)
print(f'{"quantity":14s} {"v01":>10s} {"v02":>10s} {"diff":>9s}')
for k in ('conv', 'y_gaso', 'y_lco', 'y_gas', 'y_coke', 'CO',
          'T_rgn_C', 'T_mix_C'):
    print(f'{k:14s} {r1[k]:10.4f} {r2[k]:10.4f} {r2[k]-r1[k]:+9.4f}')

ccrs = np.linspace(1.0, 9.0, 17)
rots = np.linspace(m1.ROT_LO, m1.ROT_HI, 12)
sw = {}
for tag, mod in (('v01', m1), ('v02', m2)):
    sw[tag] = dict(
        ccr=[mod.solve_unit(520.0, 220.0, 0.5, c) for c in ccrs],
        rot=[mod.solve_unit(r, 220.0, 0.5, 4.0) for r in rots])

fig, ax = plt.subplots(2, 3, figsize=(14, 8))
for j, (xs, key, xlabel) in enumerate(
        [(ccrs, 'ccr', 'feed CCR [wt%]'),
         (rots, 'rot', 'ROT setpoint [degC]')]):
    for i, (qty, ylab) in enumerate(
            [('conv', 'conversion [-]'), ('T_rgn_C', 'T_rgn [degC]'),
             ('CO', 'C/O [-]')]):
        a = ax[j, i]
        a.plot(xs, [r[qty] for r in sw['v01'][key]], 'k-', lw=2,
               label='v01 lumped')
        a.plot(xs, [r[qty] for r in sw['v02'][key]], 'r--', lw=2,
               label='v02 two-vessel')
        a.set_xlabel(xlabel); a.set_ylabel(ylab)
        a.grid(alpha=0.3)
        if i == 0 and j == 0:
            a.legend(fontsize=9)
fig.suptitle('RFCC Step 6(a) -- v01 vs v02 behaviour '
             '(CCR sweep top, ROT sweep bottom)', fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig('rfcc_step6_v01_v02_comparison.png', dpi=130,
            bbox_inches='tight')
print('saved rfcc_step6_v01_v02_comparison.png')

dmax = {}
for key, xs in (('ccr', ccrs), ('rot', rots)):
    for qty in ('conv', 'T_rgn_C', 'CO'):
        d = np.max(np.abs([a[qty] - b[qty] for a, b in
                           zip(sw['v01'][key], sw['v02'][key])]))
        dmax[f'{key}:{qty}'] = d
print('max |v02 - v01| over sweeps:',
      {k: round(v, 3) for k, v in dmax.items()})

# ---------------------------------------------------------------- (b)
print()
print('=' * 72)
print('(b) energy-balance breakdown [kJ/kg feed]')
print('=' * 72)


def print_eb(res, label):
    er, eg = res['eb_reactor'], res['eb_regen']
    print(f'--- {label} ---')
    print(f'  (T_rgn={res["T_rgn_C"]:.1f} degC, C/O={res["CO"]:.2f}, '
          f'conv={res["conv"]:.3f}, regen_cold={res["regen_cold"]})')
    print('  RISER REACTOR        IN                |  OUT')
    print(f'   cat (T_rgn)    {er["q_cat_in"]:9.1f}  | cat (T_out)  '
          f'{er["q_cat_out"]:9.1f}')
    print(f'   feed liquid    {er["q_feed_in"]:9.1f}  | oil vapour   '
          f'{er["q_oil_out"]:9.1f}')
    print(f'   steam          {er["q_steam_in"]:9.1f}  | steam        '
          f'{er["q_steam_out"]:9.1f}')
    print(f'                             | cracking duty {er["q_rxn"]:8.1f}')
    print(f'   TOTAL {er["in_total"]:9.1f}  vs  {er["out_total"]:9.1f}   '
          f'residual {er["residual"]:+8.3f} '
          f'({abs(er["residual"])/er["in_total"]*100:.4f} %)')
    print('  REGENERATOR          IN                |  OUT')
    print(f'   spent cat      {eg["q_cat_in"]:9.1f}  | regen cat    '
          f'{eg["q_cat_out"]:9.1f}')
    print(f'   coke sensible  {eg["q_coke_in"]:9.1f}  | flue gas     '
          f'{eg["q_flue_out"]:9.1f}')
    print(f'   air            {eg["q_air_in"]:9.1f}  | cat cooler   '
          f'{eg["q_cool"]:9.1f}')
    print(f'   combustion     {eg["q_comb"]:9.1f}  | wall losses  '
          f'{eg["q_loss"]:9.1f}')
    print(f'   TOTAL {eg["in_total"]:9.1f}  vs  {eg["out_total"]:9.1f}   '
          f'residual {eg["residual"]:+8.3f} '
          f'({abs(eg["residual"])/eg["in_total"]*100:.4f} %)')


res_base = m2.solve_unit(520.0, 220.0, 0.5, 4.0)
# crisis example at the baseline-trimmed operating point (at 65 % rate the
# SPECIFIC cooler duty already doubles, so qc stays at the design 0.5;
# qc = 1.0 here would over-cool the bed below self-sustained combustion)
res_heavy = m2.solve_unit(520.0, 220.0, 0.5, 6.8, 0.65)
print_eb(res_base, 'base case (CCR 4, design rate)')
print_eb(res_heavy, 'crisis heavy feed (CCR 6.8, design cooler, 65 % rate)')

# ---------------------------------------------------------------- (c)
N_SCAN = 300
B = np.array([[m1.ROT_LO, m1.ROT_HI], [m1.PRE_LO, m1.PRE_HI],
              [0.0, 1.0], [1.0, 9.0], [m1.RATE_LO, m1.RATE_HI]])
Xs = B[:, 0] + rng.uniform(size=(N_SCAN, 5)) * (B[:, 1] - B[:, 0])
res_r, res_g, cold = np.zeros(N_SCAN), np.zeros(N_SCAN), np.zeros(N_SCAN,
                                                                  bool)
for i, x in enumerate(Xs):
    r = m2.solve_unit(x[0], x[1], x[2], x[3], x[4])
    res_r[i] = r['eb_reactor']['residual']
    res_g[i] = r['eb_regen']['residual']
    cold[i] = r['regen_cold']
ok = ~cold
print()
print('=' * 72)
print(f'(c) closure-residual scan over {N_SCAN} random operating points')
print('=' * 72)
print(f'  reactor residual    : max |r| = {np.abs(res_r).max():.4f} kJ/kg'
      f'   mean |r| = {np.abs(res_r).mean():.5f} kJ/kg')
print(f'  regenerator residual: max |r| = {np.abs(res_g[ok]).max():.2e}'
      f' kJ/kg  ({ok.sum()} self-sustained points)')
print(f'  regen_cold clamped points (combustion not self-sustained, '
      f'flagged): {cold.sum()}')
if cold.sum():
    print(f'    their apparent regen residual (= clamp heat deficit): '
          f'max {np.abs(res_g[cold]).max():.1f} kJ/kg')

# ---------------------------------------------------------------- (d)
steps = [6, 10, 16, 24, 30, 60, 120]
args = (9.7, 669.6 + 273.15, 220.0 + 273.15, 4.0)
ref = RiserReactor(n_step=960).solve(*args)
errT, errQ, res_steps = [], [], []
for n in steps:
    rr = RiserReactor(n_step=n).solve(*args)
    errT.append(abs(rr['T_out_K'] - ref['T_out_K']))
    errQ.append(abs(rr['q_rxn'] - ref['q_rxn']))
    res_steps.append(abs(RiserReactor.energy_balance(rr)['residual']))
print()
print('(d) riser discretisation accuracy by step refinement '
      '(reference n = 960):')
print('   n_step   |T_out err| K   |q_rxn err| kJ/kg   '
      '|EB residual| kJ/kg')
for n, eT, eQ, r in zip(steps, errT, errQ, res_steps):
    print(f'   {n:6d}   {eT:13.3e}   {eQ:17.3e}   {r:19.3e}')
order = -np.polyfit(np.log(steps), np.log(np.maximum(errQ, 1e-16)), 1)[0]
print(f'   observed q_rxn convergence order ~ {order:.2f} '
      f'(RK4 -> 4 expected)')
print('   note: the EB residual stays at machine precision for every n --')
print('   closure is an exact bookkeeping identity of the integrator,')
print('   which is why accuracy is verified by refinement instead.')

# ---------------------------------------------------------------- (e)
print()
print('(e) operator-trim baseline policy on the v02 unit:')
for ccr, sup in [(4.0, 1.0), (6.8, 0.65), (2.5, 0.85), (8.0, 0.65)]:
    ops, r, e = m2.baseline_day(ccr, sup, m2.PRICES_BASE)
    print(f'  CCR={ccr:3.1f} sup={sup:.2f}: ROT{ops["ROT_C"]:5.0f} '
          f'pre{ops["T_pre_C"]:4.0f} qc{ops["Qcool_frac"]:.2f} '
          f'ff{ops["feed_frac"]:.2f} -> Trgn={r["T_rgn_C"]:5.0f} '
          f'conv={r["conv"]:.3f} viol=({r["viol_Trgn"]:.0f},'
          f'{r["viol_air"]:.2f},{r["viol_wgc"]:.2f}) '
          f'm={e["margin_kUSD_day"]:5.0f} k$/d')

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

# (b) energy flow bars
labels_r = ['cat in', 'feed in', 'steam in', 'cat out', 'oil out',
            'steam out', 'cracking']
labels_g = ['spent cat', 'coke sens.', 'air', 'combustion',
            'regen cat', 'flue gas', 'cooler', 'losses']
for a, res, ttl in ((ax[0, 0], res_base, 'base case (CCR 4)'),
                    (ax[0, 1], res_heavy,
                     'crisis feed (CCR 6.8, 65 % rate)')):
    er, eg = res['eb_reactor'], res['eb_regen']
    rin = [er['q_cat_in'], er['q_feed_in'], er['q_steam_in']]
    rout = [er['q_cat_out'], er['q_oil_out'], er['q_steam_out'],
            er['q_rxn']]
    gin = [eg['q_cat_in'], eg['q_coke_in'], eg['q_air_in'], eg['q_comb']]
    gout = [eg['q_cat_out'], eg['q_flue_out'], eg['q_cool'], eg['q_loss']]
    for x0, vals, lbls, cmap in ((0.0, rin, labels_r[:3], 'Blues'),
                                 (1.0, rout, labels_r[3:], 'Oranges'),
                                 (2.5, gin, labels_g[:4], 'Blues'),
                                 (3.5, gout, labels_g[4:], 'Oranges')):
        bot = 0.0
        cm = plt.get_cmap(cmap)
        for k, (v, lb) in enumerate(zip(vals, lbls)):
            a.bar(x0, v, 0.7, bottom=bot,
                  color=cm(0.35 + 0.15 * k), edgecolor='k', lw=0.4)
            if v > 250:
                a.text(x0, bot + v / 2, lb, ha='center', va='center',
                       fontsize=7)
            bot += v
    a.set_xticks([0, 1, 2.5, 3.5])
    a.set_xticklabels(['reactor\nIN', 'reactor\nOUT', 'regen\nIN',
                       'regen\nOUT'], fontsize=9)
    a.set_ylabel('enthalpy flow [kJ/kg feed]')
    a.set_title(f'(b) Energy-balance breakdown -- {ttl}')
    a.grid(alpha=0.3, axis='y')

# (c) residual scan
a = ax[1, 0]
a.semilogy(np.arange(N_SCAN)[ok], np.abs(res_r[ok]) + 1e-12, 'b.', ms=4,
           label='reactor')
a.semilogy(np.arange(N_SCAN)[ok], np.abs(res_g[ok]) + 1e-12, 'g.', ms=4,
           label='regenerator')
if cold.sum():
    a.semilogy(np.arange(N_SCAN)[cold], np.abs(res_g[cold]) + 1e-12,
               'rx', ms=5, label='regen_cold (clamped, flagged)')
a.axhline(1.0, color='k', ls=':', lw=1)
a.text(2, 1.3, '1 kJ/kg (~0.02 % of throughput)', fontsize=8)
a.set_xlabel('scan point'); a.set_ylabel('|closure residual| [kJ/kg feed]')
a.set_title(f'(c) Energy-balance closure over {N_SCAN} random points')
a.legend(fontsize=8); a.grid(alpha=0.3)

# (d) RK4 step-refinement convergence
a = ax[1, 1]
a.loglog(steps, np.maximum(errT, 1e-14), 'ko-', lw=1.5,
         label='|T_out - ref| [K]')
a.loglog(steps, np.maximum(errQ, 1e-14), 'bs-', lw=1.5,
         label='|q_rxn - ref| [kJ/kg]')
slope = errQ[0] * (np.array(steps) / steps[0]) ** -4.0
a.loglog(steps, slope, 'r--', lw=1, label='4th-order slope')
a.set_xlabel('RK4 steps along riser (reference n = 960)')
a.set_ylabel('refinement error')
a.set_title('(d) Riser discretisation accuracy:\n'
            '4th-order convergence (n = 30 in production)')
a.legend(fontsize=9); a.grid(alpha=0.3, which='both')

fig.suptitle('RFCC Step 6 -- two-vessel split (v02) energy-balance '
             'verification', fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig('rfcc_step6_energy_balance_v02.png', dpi=130,
            bbox_inches='tight')
print()
print('saved rfcc_step6_energy_balance_v02.png')
print(f'total runtime {time.time()-t_start:.1f} s')
