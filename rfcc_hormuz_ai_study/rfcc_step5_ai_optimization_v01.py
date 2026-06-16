#!/usr/bin/env python3
"""
rfcc_step5_ai_optimization_v01.py
RFCC crisis-response study -- Step 5: AI operating-condition optimisation
and the Hormuz-crisis deployment test.

Two AI strategies, both built on the Step-4 surrogate (x30,000 faster
than the physics model):

  (A) RL policy -- REINFORCE + batch baseline + Adam (pure NumPy, same
      family as the LDPE step3 RL studies in this repo).
        state  (5): [CCR, supply, feed cost, gasoline price, LCO price]
        action (4): [ROT, feed preheat, cat-cooler duty, feed rate]
        reward    : daily margin [k$/d] - equipment-constraint penalties
      Trained with domain randomisation over feed quality and prices.
      Exploration noise is scheduled (0.30 -> 0.05) and the policy mean
      is initialised at the design point.  Variance reduction uses PAIRED
      sampling: two actions per context, advantage = reward difference,
      which removes the (huge) context-to-context reward variance that a
      global batch baseline cannot.

  (B) Surrogate-MPC with digital-twin verification -- per-day random
      search (6000 LHS points on the surrogate) + Nelder-Mead polish,
      then the proposed ops are VERIFIED against the design-book ops on
      the physics model and the better one is applied ("the surrogate
      proposes, the rigorous model disposes" -- standard industrial
      practice for surrogate-based RTO).

Deployment: the 150-day Hormuz scenario is replayed on the TRUE physics
model.  All policies (baseline from Step 3 / RL / MPC) run behind the same
operator safety-trim layer, so the comparison is apples-to-apples.

Outputs: rfcc_step5_rl_training_v01.png
         rfcc_step5_crisis_response_v01.png
         rfcc_policy_v01.npz
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

from rfcc_model import (PRICES_BASE, OPEX_VAR, CAT_BASE, CAT_CCR,
                        T_PER_DAY, AFR, F_FEED_DESIGN,
                        AIR_MAX, WGC_MAX, T_RGN_MAX_C,
                        ROT_LO, ROT_HI, PRE_LO, PRE_HI, RATE_LO, RATE_HI,
                        scenario_timeline, prices_of_day, safety_trim,
                        economics)
from rfcc_surrogate import MLP

rng = np.random.default_rng(3)
t_start = time.time()
net = MLP.load('rfcc_surrogate_v01.npz')

# state normalisers (max plausible values seen in scenarios)
S_NORM = np.array([9.0, 1.05, 920.0, 1224.0, 1152.0])
A_LO = np.array([ROT_LO, PRE_LO, 0.0, RATE_LO])
A_HI = np.array([ROT_HI, PRE_HI, 1.0, RATE_HI])

# constraint-penalty weights (margin is in k$/day)
W_TRGN, W_AIR, W_WGC, W_HOLD = 3.0, 800.0, 800.0, 5.0


# =============================================================================
# A.  SURROGATE ENVIRONMENT (vectorised)
# =============================================================================
def env_reward(ops, ctx):
    """Daily margin minus constraint penalties, all on the surrogate.
    ops: (B,4) [ROT, pre, qcool, rate_raw]; ctx: dict of (B,) arrays."""
    feed_frac = np.minimum(ops[:, 3], ctx['supply'])
    X = np.column_stack([ops[:, 0], ops[:, 1], ops[:, 2],
                         ctx['CCR'], feed_frac])
    Y = net.predict(X)
    y_hs, y_lco, y_gaso, y_gas, y_coke = Y[:, 0], Y[:, 1], Y[:, 2], \
        Y[:, 3], Y[:, 4]
    T_rgn, T_out = Y[:, 5], Y[:, 7]

    f_dry = np.clip(0.15 + 0.004 * (T_out - 500.0), 0.08, 0.45)
    revenue = (y_gaso * ctx['p_gaso'] + y_lco * ctx['p_lco']
               + y_gas * (1 - f_dry) * ctx['p_lpg']
               + y_gas * f_dry * ctx['p_drygas']
               + y_hs * ctx['p_slurry'])
    cat_cost = CAT_BASE * (1.0 + CAT_CCR * ctx['CCR'])
    margin_t = revenue - ctx['p_feed'] - OPEX_VAR - cat_cost
    margin_day = margin_t * T_PER_DAY * feed_frac / 1000.0      # k$/day

    F_feed = F_FEED_DESIGN * feed_frac
    F_air = y_coke * AFR * F_feed
    F_wg = y_gas * F_feed
    pen = (W_TRGN * np.maximum(0.0, T_rgn - T_RGN_MAX_C)
           + W_AIR * np.maximum(0.0, F_air / AIR_MAX - 1.0)
           + W_WGC * np.maximum(0.0, F_wg / WGC_MAX - 1.0)
           + W_HOLD * np.maximum(0.0, (ops[:, 0] - T_out) - 3.0))
    return margin_day - pen, margin_day


def sample_context(B, rng):
    """Domain randomisation over feed quality, availability and prices."""
    b = PRICES_BASE
    return dict(
        CCR=rng.uniform(1.0, 9.0, B),
        supply=rng.uniform(0.55, 1.05, B),
        p_feed=b['feed'] * rng.uniform(0.9, 2.0, B),
        p_gaso=b['gaso'] * rng.uniform(0.9, 1.7, B),
        p_lco=b['lco'] * rng.uniform(0.9, 1.8, B),
        p_lpg=b['lpg'] * rng.uniform(0.95, 1.35, B),
        p_drygas=np.full(B, b['drygas']),
        p_slurry=b['slurry'] * rng.uniform(0.95, 1.40, B),
    )


def ctx_to_state(ctx):
    return np.column_stack([ctx['CCR'], ctx['supply'], ctx['p_feed'],
                            ctx['p_gaso'], ctx['p_lco']]) / S_NORM


def act_to_ops(A):
    """tanh-squashed action in [-1,1]^4 -> physical operating point."""
    return A_LO + 0.5 * (A + 1.0) * (A_HI - A_LO)


# =============================================================================
# B.  REINFORCE POLICY (pure NumPy)
# =============================================================================
N_S, N_A, N_H = 5, 4, 48
W1 = rng.normal(0, 0.5, (N_S, N_H)); b1 = np.zeros(N_H)
W2 = rng.normal(0, 0.3, (N_H, N_H)); b2 = np.zeros(N_H)
W3 = rng.normal(0, 0.05, (N_H, N_A))
# initialise the policy mean at the design point (tanh(b3) -> design ops)
A_DESIGN = np.array([520.0, 220.0, 0.5, 1.0])
b3 = np.arctanh(2 * (A_DESIGN - A_LO) / (A_HI - A_LO) - 1.0)
params = [W1, b1, W2, b2, W3, b3]
adam_m = [np.zeros_like(p) for p in params]
adam_v = [np.zeros_like(p) for p in params]


def policy_mu(S):
    H1 = np.tanh(S @ W1 + b1)
    H2 = np.tanh(H1 @ W2 + b2)
    return H1, H2, H2 @ W3 + b3


def policy_ops(S, sigma=0.0, rng=None):
    _, _, MU = policy_mu(np.atleast_2d(S))
    U = MU if sigma == 0.0 else MU + sigma * rng.standard_normal(MU.shape)
    return act_to_ops(np.tanh(U)), U, MU


BATCH, N_ITER = 512, 3500
SIG_HI, SIG_LO = 0.30, 0.05
val_ctx = sample_context(2048, np.random.default_rng(99))
val_S = ctx_to_state(val_ctx)

hist = {'iter': [], 'mean_R': [], 'eval_R': [], 'eval_margin': []}
print('training REINFORCE policy on the surrogate environment ...')
t = 0
for it in range(N_ITER):
    sig = SIG_HI + (SIG_LO - SIG_HI) * it / (N_ITER - 1)
    lr = 1.5e-3 if it < 0.6 * N_ITER else 5e-4
    ctx = sample_context(BATCH, rng)
    S = ctx_to_state(ctx)
    H1, H2, MU = policy_mu(S)
    # paired (antithetic) sampling: two actions on the SAME context
    Ua = MU + sig * rng.standard_normal(MU.shape)
    Ub = MU + sig * rng.standard_normal(MU.shape)
    Ra, _ = env_reward(act_to_ops(np.tanh(Ua)), ctx)
    Rb, _ = env_reward(act_to_ops(np.tanh(Ub)), ctx)
    dR = Ra - Rb                       # context variance cancels exactly
    adv = np.clip(dR / (np.abs(dR).std() + 1e-8), -3.0, 3.0)
    R = 0.5 * (Ra + Rb)                # for logging

    # gradients of -J for Adam (descent)
    G_MU = adv[:, None] * (Ua - Ub) / sig ** 2 / BATCH     # dJ/dMU
    err = -G_MU
    gW3 = H2.T @ err; gb3 = err.sum(0)
    dH2 = (err @ W3.T) * (1 - H2 * H2)
    gW2 = H1.T @ dH2; gb2 = dH2.sum(0)
    dH1 = (dH2 @ W2.T) * (1 - H1 * H1)
    gW1 = S.T @ dH1; gb1 = dH1.sum(0)
    grads = [gW1, gb1, gW2, gb2, gW3, gb3]
    t += 1
    for p, g, m, v in zip(params, grads, adam_m, adam_v):
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t))
                                          + 1e-8)

    if it % 20 == 0 or it == N_ITER - 1:
        ops_v, _, _ = policy_ops(val_S)
        R_v, M_v = env_reward(ops_v, val_ctx)
        hist['iter'].append(it)
        hist['mean_R'].append(R.mean())
        hist['eval_R'].append(R_v.mean())
        hist['eval_margin'].append(M_v.mean())
        if it % 200 == 0:
            print(f'  iter {it:5d}  batch R {R.mean():7.1f}   '
                  f'greedy eval R {R_v.mean():7.1f} k$/d   '
                  f'sigma {sig:.3f}')

np.savez('rfcc_policy_v01.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
         S_NORM=S_NORM, A_LO=A_LO, A_HI=A_HI)
print('saved rfcc_policy_v01.npz')

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(hist['iter'], hist['mean_R'], alpha=0.5, label='batch (noisy)')
ax[0].plot(hist['iter'], hist['eval_R'], 'r-', lw=2, label='greedy eval')
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('reward [k$/day]')
ax[0].set_title('(a) REINFORCE training (surrogate env,\n'
                'domain-randomised feed & prices)')
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(hist['iter'], hist['eval_margin'], 'g-', lw=2)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('margin [k$/day]')
ax[1].set_title('(b) Greedy evaluation margin (before penalties)')
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig('rfcc_step5_rl_training_v01.png', dpi=130, bbox_inches='tight')
print('saved rfcc_step5_rl_training_v01.png')


# =============================================================================
# C.  SURROGATE-MPC (per-day Nelder-Mead multistart)
# =============================================================================
MPC_RNG = np.random.default_rng(11)


def mpc_ops(ctx1, n_rand=6000):
    """Optimise one day's ops on the surrogate: LHS random search over the
    whole action box, then Nelder-Mead polish from the top-3 candidates."""
    ctxB = {k: np.full(n_rand, v) for k, v in ctx1.items()}
    Z = A_LO + MPC_RNG.uniform(size=(n_rand, 4)) * (A_HI - A_LO)
    R, _ = env_reward(Z, ctxB)
    top = Z[np.argsort(R)[-3:]]

    def neg_r(z):
        zc = np.clip(z, A_LO, A_HI)[None, :]
        r, _ = env_reward(zc, {k: np.atleast_1d(v)
                               for k, v in ctx1.items()})
        return -float(r[0]) + 1e3 * np.sum(np.abs(z - zc[0]))

    best, best_f = None, np.inf
    for s0 in top:
        r = minimize(neg_r, s0, method='Nelder-Mead',
                     options=dict(maxiter=300, xatol=1e-2, fatol=1e-2))
        if r.fun < best_f:
            best, best_f = np.clip(r.x, A_LO, A_HI), r.fun
    return best


# =============================================================================
# D.  DEPLOYMENT: 150-DAY CRISIS REPLAY ON THE TRUE PHYSICS MODEL
# =============================================================================
print('deploying policies on the physics model over the crisis timeline ...')
scen = scenario_timeline(n_days=150, seed=42)
base = np.load('rfcc_step3_baseline_v01.npz')
days = scen['days']; n_days = len(days)

POL = ['RL', 'MPC']
dep = {p: {k: np.zeros(n_days) for k in
           ['margin', 'ROT_C', 'T_pre_C', 'Qcool_frac', 'feed_frac',
            'T_rgn_C', 'conv']} for p in POL}
n_mpc_accept = 0

for d in days:
    prices = prices_of_day(scen, d)
    ctx1 = dict(CCR=scen['CCR'][d], supply=scen['supply'][d],
                p_feed=prices['feed'], p_gaso=prices['gaso'],
                p_lco=prices['lco'], p_lpg=prices['lpg'],
                p_drygas=prices['drygas'], p_slurry=prices['slurry'])
    s = np.array([ctx1['CCR'], ctx1['supply'], ctx1['p_feed'],
                  ctx1['p_gaso'], ctx1['p_lco']]) / S_NORM

    for pol in POL:
        if pol == 'RL':
            o, _, _ = policy_ops(s)
            o = o[0]
        else:
            o = mpc_ops(ctx1)
        ops = dict(ROT_C=o[0], T_pre_C=o[1], Qcool_frac=o[2],
                   feed_frac=min(o[3], ctx1['supply']))
        ops_t, res, eco = safety_trim(ops, ctx1['CCR'], prices)
        if pol == 'MPC':
            # digital-twin verification: keep the surrogate proposal only
            # if the physics model confirms it beats the design-book ops
            ops_b = dict(ROT_C=520.0, T_pre_C=220.0, Qcool_frac=0.5,
                         feed_frac=min(1.0, ctx1['supply']))
            ops_bt, res_b, eco_b = safety_trim(ops_b, ctx1['CCR'], prices)
            if eco['margin_kUSD_day'] >= eco_b['margin_kUSD_day']:
                n_mpc_accept += 1
            else:
                ops_t, res, eco = ops_bt, res_b, eco_b
        for k in ('ROT_C', 'T_pre_C', 'Qcool_frac', 'feed_frac'):
            dep[pol][k][d] = ops_t[k]
        dep[pol]['margin'][d] = eco['margin_kUSD_day']
        dep[pol]['T_rgn_C'][d] = res['T_rgn_C']
        dep[pol]['conv'][d] = res['conv']
    if d % 25 == 0:
        print(f'  day {d:3d}  base {base["margin_kUSD_day"][d]:7.0f}  '
              f'RL {dep["RL"]["margin"][d]:7.0f}  '
              f'MPC {dep["MPC"]["margin"][d]:7.0f} k$/d')
print(f'MPC proposal accepted by twin verification on '
      f'{n_mpc_accept}/{n_days} days')

cum_b = np.cumsum(base['margin_kUSD_day']) / 1e3
cum_rl = np.cumsum(dep['RL']['margin']) / 1e3
cum_mpc = np.cumsum(dep['MPC']['margin']) / 1e3

# ---------------------------------------------------------------- plots
PHASE_EDGES = [0, 15, 35, 70, 110, 150]
PHASE_COLORS = ['#ffffff', '#ffcccc', '#fff2cc', '#e2efda', '#ffffff']


def shade(ax):
    for k in range(5):
        ax.axvspan(PHASE_EDGES[k], PHASE_EDGES[k + 1],
                   color=PHASE_COLORS[k], alpha=0.5, zorder=0)


fig, ax = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
a = ax[0]; shade(a)
a.plot(days, base['margin_kUSD_day'], 'k-', lw=1.8, label='baseline')
a.plot(days, dep['RL']['margin'], 'r-', lw=1.8, label='AI: RL policy')
a.plot(days, dep['MPC']['margin'], 'b--', lw=1.5, label='AI: surrogate-MPC')
a.set_ylabel('margin [k$/day]')
a.set_title('Hormuz crisis deployment on the physics model -- daily margin')
a.legend(fontsize=9); a.grid(alpha=0.3)

a = ax[1]; shade(a)
a.plot(days, cum_rl - cum_b, 'r-', lw=2,
       label=f'RL  - baseline  (final {cum_rl[-1]-cum_b[-1]:+.1f} M$)')
a.plot(days, cum_mpc - cum_b, 'b--', lw=2,
       label=f'MPC - baseline  (final {cum_mpc[-1]-cum_b[-1]:+.1f} M$)')
a.axhline(0, color='k', lw=0.8)
a.set_ylabel('cumulative advantage [M$]')
a.set_title('Cumulative margin advantage of AI operation')
a.legend(fontsize=9); a.grid(alpha=0.3)

a = ax[2]; shade(a)
a.plot(days, base['ROT_C'], 'k-', lw=1.5, label='ROT baseline')
a.plot(days, dep['RL']['ROT_C'], 'r-', lw=1.5, label='ROT RL')
a.set_ylabel('ROT [degC]')
a2 = a.twinx()
a2.plot(days, base['feed_frac'] * 100, 'k:', lw=1.2)
a2.plot(days, dep['RL']['feed_frac'] * 100, 'r:', lw=1.5)
a2.plot(days, dep['RL']['Qcool_frac'] * 100, color='orange', lw=1.2,
        label='cooler RL')
a2.set_ylabel('feed rate (dotted) / cooler [%]')
a2.legend(fontsize=8, loc='lower right')
a.set_title('Operating conditions: proactive AI vs reactive baseline')
a.legend(fontsize=9, loc='upper right'); a.grid(alpha=0.3)

a = ax[3]; shade(a)
a.plot(days, base['T_rgn_C'], 'k-', lw=1.5, label='baseline')
a.plot(days, dep['RL']['T_rgn_C'], 'r-', lw=1.5, label='RL')
a.axhline(T_RGN_MAX_C, color='r', ls='--', lw=1.2, label='limit')
a.set_ylabel('T_rgn [degC]'); a.set_xlabel('day')
a.set_title('Regenerator temperature')
a.legend(fontsize=9); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('rfcc_step5_crisis_response_v01.png', dpi=130,
            bbox_inches='tight')
print('saved rfcc_step5_crisis_response_v01.png')

# ---------------------------------------------------------------- summary
print()
print('=' * 68)
print('150-day Hormuz crisis -- cumulative margins (physics model)')
print('=' * 68)
print(f"  baseline (design book + operator trims): {cum_b[-1]:8.1f} M$")
print(f"  AI RL policy                           : {cum_rl[-1]:8.1f} M$  "
      f"(+{cum_rl[-1]-cum_b[-1]:.1f})")
print(f"  AI surrogate-MPC                       : {cum_mpc[-1]:8.1f} M$  "
      f"(+{cum_mpc[-1]-cum_b[-1]:.1f})")
PHASE_NAMES = ['normal', 'blockade shock', 'adaptation', 'partial reopen',
               'normalisation']
for k in range(5):
    sl = slice(PHASE_EDGES[k], PHASE_EDGES[k + 1])
    print(f"  phase {k} {PHASE_NAMES[k]:16s}: base "
          f"{base['margin_kUSD_day'][sl].mean():7.0f}  RL "
          f"{dep['RL']['margin'][sl].mean():7.0f}  MPC "
          f"{dep['MPC']['margin'][sl].mean():7.0f}  k$/day")
print(f'total runtime {time.time()-t_start:.1f} s')
