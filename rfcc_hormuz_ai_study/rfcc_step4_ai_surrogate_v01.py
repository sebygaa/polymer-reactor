#!/usr/bin/env python3
"""
rfcc_step4_ai_surrogate_v01.py
RFCC crisis-response study -- Step 4: neural-network surrogate model.

  (1) Latin-Hypercube sample the operating + feed space and label it with
      the physics model (rfcc_model.solve_unit)
  (2) train a pure-NumPy MLP surrogate (5 -> 96 -> 96 -> 8)
  (3) validate (parity plots, R2 per output) and benchmark the speed-up

The surrogate is the fast environment used by the AI optimisers (Step 5);
economics and constraint slacks are computed analytically from its outputs.

Outputs: rfcc_dataset_v01.npz, rfcc_surrogate_v01.npz,
         rfcc_step4_surrogate_v01.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

from rfcc_model import (solve_unit, ROT_LO, ROT_HI, PRE_LO, PRE_HI,
                        RATE_LO, RATE_HI)
from rfcc_surrogate import MLP, X_NAMES, Y_NAMES

N_TRAIN, N_TEST = 3500, 700
BOUNDS = np.array([[ROT_LO, ROT_HI],      # ROT_C
                   [PRE_LO, PRE_HI],      # T_pre_C
                   [0.0, 1.0],            # Qcool_frac
                   [1.0, 9.0],            # CCR
                   [RATE_LO, RATE_HI]])   # feed_frac

rng = np.random.default_rng(7)


def lhs(n, bounds, rng):
    """Simple Latin-Hypercube sample."""
    d = len(bounds)
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T
         + rng.uniform(size=(n, d))) / n
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def label(X):
    Y = np.zeros((len(X), len(Y_NAMES)))
    t0 = time.time()
    for i, x in enumerate(X):
        r = solve_unit(x[0], x[1], x[2], x[3], x[4])
        Y[i] = [r['y_hs'], r['y_lco'], r['y_gaso'], r['y_gas'], r['y_coke'],
                r['T_rgn_C'], r['CO'], r['T_out_C']]
        if i % 500 == 0:
            print(f'  labelled {i}/{len(X)}  ({time.time()-t0:.0f} s)')
    return Y


# ---------------------------------------------------------------- (1)
if os.path.exists('rfcc_dataset_v01.npz'):
    d = np.load('rfcc_dataset_v01.npz')
    X_tr, Y_tr, X_te, Y_te = d['X_tr'], d['Y_tr'], d['X_te'], d['Y_te']
    print(f'loaded cached dataset ({len(X_tr)} train / {len(X_te)} test)')
else:
    print(f'generating dataset: {N_TRAIN} train + {N_TEST} test points ...')
    X_tr = lhs(N_TRAIN, BOUNDS, rng)
    X_te = lhs(N_TEST, BOUNDS, rng)
    Y_tr = label(X_tr)
    Y_te = label(X_te)
    np.savez('rfcc_dataset_v01.npz', X_tr=X_tr, Y_tr=Y_tr,
             X_te=X_te, Y_te=Y_te)
    print('saved rfcc_dataset_v01.npz')

# ---------------------------------------------------------------- (2)
print('training MLP surrogate (5 -> 96 -> 96 -> 8) ...')
net = MLP(len(X_NAMES), len(Y_NAMES), n_hid=96, seed=0)
hist = net.fit(X_tr, Y_tr, X_te, Y_te, epochs=1500, batch=256, lr=1e-3,
               verbose_every=300)
net.save('rfcc_surrogate_v01.npz')
print('saved rfcc_surrogate_v01.npz')

# ---------------------------------------------------------------- (3)
Y_hat = net.predict(X_te)
r2 = 1 - ((Y_te - Y_hat) ** 2).sum(0) / \
    ((Y_te - Y_te.mean(0)) ** 2).sum(0)
print('test R2 per output:')
for n_, r_ in zip(Y_NAMES, r2):
    print(f'  {n_:8s} {r_:.4f}')

# speed benchmark
t0 = time.time()
for x in X_te[:50]:
    solve_unit(x[0], x[1], x[2], x[3], x[4])
t_sim = (time.time() - t0) / 50
t0 = time.time()
for _ in range(20):
    net.predict(X_te)            # 700 points per call
t_sur = (time.time() - t0) / (20 * len(X_te))
print(f'physics model : {t_sim*1e3:8.2f} ms/eval')
print(f'surrogate     : {t_sur*1e3:8.4f} ms/eval  '
      f'(speed-up x{t_sim/t_sur:,.0f})')

# plots: loss curve + parity for 5 key outputs
fig, ax = plt.subplots(2, 3, figsize=(14, 8.5))
a = ax[0, 0]
a.semilogy(hist['epoch'], hist['train'], 'b-', label='train')
a.semilogy(hist['epoch'], hist['val'], 'r-', label='test')
a.set_xlabel('epoch'); a.set_ylabel('MSE (standardised)')
a.set_title('(a) Training history')
a.legend(); a.grid(alpha=0.3)

for a, (idx, lbl) in zip(
        [ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]],
        [(2, 'gasoline yield [-]'), (4, 'coke yield [-]'),
         (5, 'T_rgn [degC]'), (6, 'C/O [-]'), (1, 'LCO yield [-]')]):
    a.plot(Y_te[:, idx], Y_hat[:, idx], '.', ms=3, alpha=0.4)
    lim = [Y_te[:, idx].min(), Y_te[:, idx].max()]
    a.plot(lim, lim, 'k--', lw=1)
    a.set_xlabel('physics model'); a.set_ylabel('surrogate')
    a.set_title(f'{lbl}   R2 = {r2[idx]:.4f}')
    a.grid(alpha=0.3)

fig.suptitle('RFCC Step 4 -- NumPy MLP surrogate of the heat-balanced unit',
             fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig('rfcc_step4_surrogate_v01.png', dpi=130, bbox_inches='tight')
print('saved rfcc_step4_surrogate_v01.png')
