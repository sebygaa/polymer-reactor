#!/usr/bin/env python3
"""
step3_rl_optimization_v05.py
LDPE Tubular PFR — RL Operating Condition Optimisation  v05

Changes from v04 (step3_rl_optimization_v04.py):
  v05-1  β-scission added to the fast PFR simulator, synced with the physical
         model in step2_rxn_w_T_effect_v04.py.

         Mechanism (symmetric random-scission approximation):
           Pₙ  →  P_{n/2}  +  D_{n/2}          rate = k_bs · λ₀

         Moment contributions:
           dλ₁/dt|_bs = −k_bs · λ₁/2            live radicals shorten
           dλ₂/dt|_bs = −k_bs · λ₂ · 2/3
           dμ₀/dt|_bs = +k_bs · λ₀              one dead fragment per event
           dμ₁/dt|_bs = +k_bs · λ₁/2
           dμ₂/dt|_bs = +k_bs · λ₂/3

         Parameters (effective lumped; no explicit mid-chain radical species):
           A_kbs  = 1.0e11  s⁻¹     Ea_kbs = 130 000  J/mol

         Effect on k_bs:
           195 °C → ≈ 3.8e-4 s⁻¹ (negligible)
           250 °C → ≈ 1.1e-2 s⁻¹ (minor, ~4 % Mn drop in hot-spot zone)
           280 °C → ≈ 4.9e-2 s⁻¹ (notable, ~10 % Mn drop)
           300 °C → ≈ 1.3e-1 s⁻¹ (significant)

         Physical motivation: at T_peak > 250 °C, mid-chain radicals formed
         by ktrp undergo β-scission, fragmenting the chain and decreasing Mn.
         This is important when the RL explores high-T operating conditions.

Inherited from v04:
  v04-1  A_ktrm = 1550 (synced with step2_v03; was 1.5 in v03 RL)
  v04-2  PDI reward redesigned for single-zone MOM reality [1.5–2.5]
  v04-3  Warm start: episodes begin from previous episode's final OC;
         re-randomisation every N_RESET = 25 episodes

Algorithm  : REINFORCE + moving-average baseline + Adam
Policy     : 2-hidden-layer MLP (pure NumPy)
Environment: Fast LDPE PFR (N=20, Radau, loose tolerance)

State  (8-D): [X, T_peak*, Mn*, PDI*, T0*, log_ini*, Tc*, U*]
Action (4-D): [δT₀, δlog(ini₀), δTc, δU]  tanh-squashed

Physical model: L=1500 m, v=10 m/s → τ_res=150 s  (full 3-zone reactor)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import time, warnings

np.random.seed(0)

# =============================================================================
# A.  PHYSICAL & KINETIC CONSTANTS
#     v04-1: A_ktrm = 1550 (was 1.5 in v03 RL)
#     v05-1: A_kbs, Ea_kbs added
# =============================================================================
R_gas   = 8.3145
Mw_mono = 28.054
f_eff   = 0.8
D       = 0.05
D_j     = 0.07
A_c     = np.pi / 4 * (D_j**2 - D**2)
v_flow  = 10.0
v_cool  = 1.0
rho     = 600_000.0
Cp      = 1.7
dH_p    = -93_000.0
rho_c   = 900_000.0
Cp_c    = 4.18
mono_0  = 2.005e4
L_react = 1500.0
NV      = 10

# Arrhenius — synced with step2_rxn_w_T_effect_v04.py
A_kd   = 3.15e15;  Ea_kd   = 155_000.0
A_kp   = 6.58e4;   Ea_kp   =  29_500.0
A_ktc  = 2.0e5;    Ea_ktc  =   5_000.0
A_ktd  = 2.0e5;    Ea_ktd  =   5_000.0
A_ktrm = 1550.;    Ea_ktrm =  47_000.0   # v04-1: corrected from 1.5
A_ktrp = 3.0e-1;   Ea_ktrp =  50_000.0
A_kbs  = 1.0e11;   Ea_kbs  = 130_000.0  # v05-1: β-scission

def arrh(A, Ea, T):
    return A * np.exp(-Ea / (R_gas * T))

def _build_jac(N):
    rows, cols = [], []
    for i in range(N):
        for j in range(NV):
            r = i * NV + j
            for k in range(NV):
                rows.append(r); cols.append(i * NV + k)
            if j < 9 and i > 0:
                rows.append(r); cols.append((i-1)*NV + j)
            if j == 9 and i < N - 1:
                rows.append(r); cols.append((i+1)*NV + 9)
    return csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(N*NV, N*NV)
    )

# =============================================================================
# B.  FAST LDPE PFR SIMULATOR  (v05-1: β-scission terms added)
# =============================================================================
def run_pfr(T_0, ini_0, Tc_in, U_heat, N=20, rtol=5e-3, t_frac=1.8):
    """
    Integrate LDPE PFR moment ODEs to pseudo-steady state.
    Returns (X_exit, T_peak_K, Mn_g/mol, Mw_g/mol, PDI, success).
    """
    h_r = 4.0 * U_heat / (rho * Cp * D)
    h_j = U_heat * np.pi * D / (rho_c * Cp_c * A_c)
    dz  = L_react / (N - 1)
    tau = L_react / v_flow
    jac = _build_jac(N)
    atol = np.tile([1e-12, 1e-9, 1e-5, 1e-10, 1e-7, 1e-2, 1e-7, 1., 1e-3, 1e-3], N)

    def odes(t, y):
        s = y.reshape(N, NV)
        l0, l1, l2 = s[:,0], s[:,1], s[:,2]
        m0, m1, m2 = s[:,3], s[:,4], s[:,5]
        ini, mono, T, Tc = s[:,6], s[:,7], s[:,8], s[:,9]

        kd   = arrh(A_kd,   Ea_kd,   T)
        kp   = arrh(A_kp,   Ea_kp,   T)
        ktc  = arrh(A_ktc,  Ea_ktc,  T)
        ktd  = arrh(A_ktd,  Ea_ktd,  T)
        ktrm = arrh(A_ktrm, Ea_ktrm, T)
        ktrp = arrh(A_ktrp, Ea_ktrp, T)
        kbs  = arrh(A_kbs,  Ea_kbs,  T)   # v05-1

        eps  = 1e-12
        mu3h = np.where((m0 > eps) & (m1 > eps),
                        m2*(2*m0*m2 - m1**2) / (m0*m1 + eps), 0.)
        mu3  = np.maximum(mu3h, np.where(m1 > eps, m2**2 / (m1 + eps), 0.))

        # v05-1: β-scission  Pₙ → P_{n/2} + D_{n/2}  (symmetric split)
        #   dλ₀|_bs = 0  (radical count conserved)
        #   dλ₁|_bs = −k_bs·λ₁/2
        #   dλ₂|_bs = −k_bs·λ₂·2/3
        #   dμ₀|_bs = +k_bs·λ₀
        #   dμ₁|_bs = +k_bs·λ₁/2
        #   dμ₂|_bs = +k_bs·λ₂/3
        Rl0 = 2*f_eff*kd*ini - (ktc+ktd)*l0**2
        Rl1 = (kp*mono*l0 + ktrm*mono*(l0-l1)
               + ktrp*(l0*m2-l1*m1) - (ktc+ktd)*l0*l1
               - kbs*l1/2.)
        Rl2 = (kp*mono*(2*l1+l0) + ktrm*mono*(l0-l2)
               + ktrp*(l0*mu3-l2*m1) - (ktc+ktd)*l0*l2
               - kbs*l2*2./3.)
        Rm0 = (ktrm*mono*l0 + (0.5*ktc+ktd)*l0**2
               + kbs*l0)
        Rm1 = (ktrm*mono*l1 + ktrp*(l1*m1-l0*m2) + (ktc+ktd)*l0*l1
               + kbs*l1/2.)
        Rm2 = (ktrm*mono*l2 + ktd*l0*l2
               + ktc*(l0*l2+l1**2) + ktrp*(l2*m1-l0*mu3)
               + kbs*l2/3.)
        Ri  = -kd*ini
        Rm  = -kp*mono*l0
        RT  = (-dH_p)/(rho*Cp)*kp*mono*l0 - h_r*(T - Tc)
        T_safe = 320. + 273.15
        RT -= np.maximum(0., (T - T_safe)) * 20.0
        RTc = h_j*(T - Tc)

        dy = np.zeros_like(s)
        for j, (C, R, BC) in enumerate(zip(
            [l0, l1, l2, m0, m1, m2, ini, mono, T],
            [Rl0,Rl1,Rl2,Rm0,Rm1,Rm2, Ri,  Rm,  RT],
            [0., 0., 0., 0., 0., 0., ini_0, mono_0, T_0]
        )):
            Cu = np.empty(N); Cu[0] = BC; Cu[1:] = C[:-1]
            dy[0, j] = 0.
            dy[1:, j] = R[1:] - v_flow*(C[1:] - Cu[1:])/dz

        Cd = np.empty(N); Cd[:-1] = Tc[1:]; Cd[-1] = Tc_in
        dy[:N-1, 9] = RTc[:N-1] + v_cool*(Cd[:N-1] - Tc[:N-1])/dz
        dy[N-1,  9] = 0.
        return dy.ravel()

    y0 = np.zeros((N, NV))
    y0[:, 7] = mono_0; y0[:, 8] = T_0; y0[:, 9] = Tc_in
    y0[0,  6] = ini_0; y0[N-1, 9] = Tc_in

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sol = solve_ivp(
                odes, (0., t_frac*tau), y0.ravel(),
                method='Radau', t_eval=[t_frac*tau],
                rtol=rtol, atol=atol,
                jac_sparsity=jac, max_step=tau/5
            )
        if sol.status != 0:
            return 0., 9999., 0., 0., 0., False

        Y   = sol.y.T.reshape(1, N, NV)
        mf  = Y[0,:,7]; Tf  = Y[0,:,8]
        m0f = Y[0,:,3]; m1f = Y[0,:,4]; m2f = Y[0,:,5]
        eps = 1e-30
        X   = float(np.clip(1. - mf[-1]/mono_0, 0., 1.))
        Tpk = float(Tf.max())
        Mn  = float(Mw_mono*(m1f[-1]+eps)/(m0f[-1]+eps))
        Mw  = float(Mw_mono*(m2f[-1]+eps)/(m1f[-1]+eps))
        PDI = Mw / max(Mn, 1.)
        if Tpk > 1500. or np.isnan(X) or np.isnan(Tpk):
            return 0., Tpk, Mn, Mw, PDI, False
        return X, Tpk, Mn, Mw, PDI, True
    except Exception:
        return 0., 9999., 0., 0., 0., False

# =============================================================================
# C.  REWARD  (v04-2: PDI target [1.5–2.5] for single-zone MOM)
# =============================================================================
def compute_reward(X, Tpk, Mn, Mw, PDI, ok):
    """
    Multi-objective reward for LDPE single-injection-zone.
    v04-2: PDI target [1.5–2.5] (single-zone MOM reality; old [3–15] unreachable).
    v05-1: β-scission in simulator means high-T states now produce lower Mn,
           which is physically consistent and feeds correctly into the Mn reward.
    """
    if not ok:
        return -100.

    r = 0.
    Tc   = Tpk - 273.15
    Mn_k = Mn  / 1000.
    Mw_k = Mw  / 1000.

    # --- Conversion (zone target 5–15 %, centre 10 %) ---
    if 0.05 <= X <= 0.15:
        r += 15. * (1. - abs(X - 0.10) / 0.05)
    elif X < 0.05:
        r -= 20. * (0.05 - X) / 0.05
    else:
        r -= 15. * (X - 0.15) / 0.10

    # --- Peak temperature ---
    if 200. <= Tc <= 300.:
        r += 12.
    elif Tc < 200.:
        r -= 5. * (200. - Tc) / 50.
    elif Tc <= 320.:
        r -= 15. * (Tc - 300.) / 20.
    else:
        r -= 50. + 0.5*(Tc - 320.)

    # --- Mn ---
    if 30. <= Mn_k <= 150.:
        r += 8.
    else:
        r -= 5. * max(30. - Mn_k, Mn_k - 150.) / 60.

    # --- Mw ---
    if 100. <= Mw_k <= 500.:
        r += 8.
    else:
        r -= 4. * max(100. - Mw_k, Mw_k - 500.) / 200.

    # --- PDI  (v04-2) ---
    # Single-zone MOM gives PDI ≈ 1.7–2.1; old target [3–15] was unreachable.
    if 1.5 <= PDI <= 2.5:
        r += 5.
    elif PDI < 1.5:
        r -= 3.
    elif PDI <= 4.0:
        pass
    else:
        r -= 2. * (PDI - 4.0) / 4.0

    return float(r)

# =============================================================================
# D.  OPERATING CONDITION SPACE
# =============================================================================
OC_LO = np.array([413.15, np.log(0.003), 363.15,  400.])
OC_HI = np.array([483.15, np.log(0.100), 443.15, 3000.])
OC_SC = np.array([8.,     0.30,           8.,    250.])

def oc_clip(oc):    return np.clip(oc, OC_LO, OC_HI)
def oc_to_real(oc): r = oc.copy(); r[1] = np.exp(oc[1]); return r
def oc_norm(oc):    return (oc - OC_LO) / (OC_HI - OC_LO) * 2. - 1.
def random_oc():    return OC_LO + np.random.rand(4) * (OC_HI - OC_LO)

def build_state(X, Tpk, Mn, PDI, oc):
    s = np.array([
        X / 0.40,
        (Tpk - 273.15) / 400.,
        Mn  / 6e4,
        PDI / 25.,
        *oc_norm(oc)
    ], dtype=np.float64)
    return np.clip(s, -3., 3.)

# =============================================================================
# E.  GAUSSIAN MLP POLICY  (pure NumPy)
# =============================================================================
class GaussianMLP:
    def __init__(self, s_dim=8, h_dim=64, a_dim=4, lr=4e-3, seed=7):
        rng = np.random.default_rng(seed)
        k = lambda i: np.sqrt(2./i)
        self.W1  = rng.normal(0, k(s_dim), (h_dim, s_dim))
        self.b1  = np.zeros(h_dim)
        self.W2  = rng.normal(0, k(h_dim), (h_dim, h_dim))
        self.b2  = np.zeros(h_dim)
        self.W3  = rng.normal(0, k(h_dim), (a_dim, h_dim))
        self.b3  = np.zeros(a_dim)
        self.lst = np.full(a_dim, -1.0)
        self.lr  = lr
        self.t   = 0
        p = self._flat()
        self.m  = np.zeros_like(p)
        self.va = np.zeros_like(p)

    def forward(self, s):
        self._s  = s
        self._h1 = np.tanh(self.W1 @ s + self.b1)
        self._h2 = np.tanh(self.W2 @ self._h1 + self.b2)
        self._mu = self.W3 @ self._h2 + self.b3
        self._ls = np.clip(self.lst, -3., 0.)
        return self._mu, self._ls

    def sample(self, s):
        mu, ls = self.forward(s)
        eps = np.random.randn(len(mu))
        u   = mu + np.exp(ls) * eps
        a   = np.tanh(u)
        lp  = ((-0.5*eps**2 - ls - 0.5*np.log(2*np.pi)).sum()
               - np.log(1. - a**2 + 1e-6).sum())
        return a, u, lp

    def greedy(self, s):
        mu, _ = self.forward(s)
        return np.tanh(mu)

    def pg_grad(self, s, u, advantage):
        mu, ls = self.forward(s)
        std    = np.exp(ls)
        d_mu   = advantage * (u - mu) / std**2
        d_lst  = advantage * ((u - mu)**2 / std**2 - 1.)
        dW3 = np.outer(d_mu, self._h2);  db3 = d_mu
        dh2 = self.W3.T @ d_mu
        d2p = dh2 * (1. - self._h2**2)
        dW2 = np.outer(d2p, self._h1);   db2 = d2p
        dh1 = self.W2.T @ d2p
        d1p = dh1 * (1. - self._h1**2)
        dW1 = np.outer(d1p, self._s);    db1 = d1p
        return np.concatenate([g.ravel() for g in
               [dW1, db1, dW2, db2, dW3, db3, d_lst]])

    def update(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.m  = b1*self.m  + (1-b1)*grads
        self.va = b2*self.va + (1-b2)*grads**2
        mh = self.m  / (1 - b1**self.t)
        vh = self.va / (1 - b2**self.t)
        delta = self.lr * mh / (np.sqrt(vh) + eps)
        idx = 0; new = []
        for p in self._params():
            n = p.size
            new.append(p + delta[idx:idx+n].reshape(p.shape))
            idx += n
        (self.W1, self.b1, self.W2, self.b2,
         self.W3, self.b3, self.lst) = new

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3, self.lst]

    def _flat(self):
        return np.concatenate([p.ravel() for p in self._params()])

# =============================================================================
# F.  REINFORCE TRAINING LOOP
#     v04-3: warm start — each episode begins from previous episode's final OC.
#            Re-randomisation every N_RESET = 25 episodes.
# =============================================================================
N_EPS   = 150
N_STEPS = 6
N_RESET = 25
GAMMA   = 0.95
EMA_A   = 0.10

policy       = GaussianMLP()
ep_rewards   = []
ep_best_log  = []
ep_was_reset = []
baseline     = 0.
best_r       = -1e9
best_oc      = None
best_result  = None

print("=" * 70)
print("  LDPE PFR — RL Optimisation  v05  (REINFORCE + warm start + β-scission)")
print("=" * 70)
print(f"  v04-1  A_ktrm = 1550  (corrected from 1.5)")
print(f"  v04-2  PDI reward: target [1.5–2.5] (single-zone MOM)")
print(f"  v04-3  Warm start from previous episode's final OC")
print(f"  v05-1  β-scission in simulator: A_kbs=1e11, Ea_kbs=130 kJ/mol")
print(f"         k_bs(195°C)≈3.8e-4 s⁻¹  k_bs(250°C)≈1.1e-2  k_bs(280°C)≈4.9e-2")
print(f"  Episodes : {N_EPS} × {N_STEPS} steps = {N_EPS*N_STEPS} simulations (N=20)")
rr_lo = oc_to_real(OC_LO); rr_hi = oc_to_real(OC_HI)
print(f"  OC search space:")
print(f"    T₀      : {rr_lo[0]-273.15:.0f} – {rr_hi[0]-273.15:.0f} °C")
print(f"    ini₀    : {rr_lo[1]:.4f} – {rr_hi[1]:.3f} mol/m³")
print(f"    Tc_in   : {rr_lo[2]-273.15:.0f} – {rr_hi[2]-273.15:.0f} °C")
print(f"    U_heat  : {rr_lo[3]:.0f} – {rr_hi[3]:.0f} W/(m²·K)")
print("=" * 70)
t0_train = time.time()

oc = random_oc()

for ep in range(N_EPS):
    rerandom = (ep > 0 and ep % N_RESET == 0)
    if rerandom:
        oc = random_oc()
    ep_was_reset.append(rerandom)

    r_oc = oc_to_real(oc)
    X, Tpk, Mn, Mw, PDI, ok = run_pfr(r_oc[0], r_oc[1], r_oc[2], r_oc[3])
    s = build_state(X, Tpk, Mn, PDI, oc)

    traj    = []
    ep_rtot = 0.

    for step in range(N_STEPS):
        a, u, lp = policy.sample(s)
        oc_new   = oc_clip(oc + a * OC_SC)
        r_oc2    = oc_to_real(oc_new)
        X2, Tpk2, Mn2, Mw2, PDI2, ok2 = run_pfr(
            r_oc2[0], r_oc2[1], r_oc2[2], r_oc2[3]
        )
        r_step = compute_reward(X2, Tpk2, Mn2, Mw2, PDI2, ok2)
        traj.append((s.copy(), u.copy(), a.copy(), r_step))
        ep_rtot += r_step

        if ok2 and r_step > best_r:
            best_r      = r_step
            best_oc     = oc_new.copy()
            best_result = (X2, Tpk2, Mn2, Mw2, PDI2)

        s  = build_state(X2, Tpk2, Mn2, PDI2, oc_new)
        oc = oc_new

    G = 0.; returns = []
    for (_, _, _, ri) in reversed(traj):
        G = ri + GAMMA * G; returns.insert(0, G)
    returns  = np.array(returns)
    baseline = (1 - EMA_A)*baseline + EMA_A*returns.mean()
    advs     = returns - baseline

    total_g = np.zeros_like(policy._flat())
    for i, (si, ui, ai, _) in enumerate(traj):
        total_g += policy.pg_grad(si, ui, advs[i])
    policy.update(total_g / N_STEPS)

    ep_rewards.append(ep_rtot)
    ep_best_log.append(best_r)

    if (ep + 1) % 10 == 0 or ep == 0:
        el  = time.time() - t0_train
        tag = " [re-rand]" if rerandom else " [warm]   "
        print(f"  Ep {ep+1:3d}/{N_EPS}{tag}  ep_r={ep_rtot:+7.1f}  "
              f"best={best_r:+6.1f}  t={el:.0f}s")
        if best_result:
            Xb, Tpb, Mnb, Mwb, Pb = best_result
            rr = oc_to_real(best_oc)
            print(f"    best OC : T₀={rr[0]-273.15:.1f}°C  ini₀={rr[1]:.4f}  "
                  f"Tc={rr[2]-273.15:.1f}°C  U={rr[3]:.0f}")
            print(f"    results : X={Xb*100:.1f}%  Tpk={Tpb-273.15:.1f}°C  "
                  f"Mn={Mnb/1e3:.1f} kg/mol  Mw={Mwb/1e3:.1f} kg/mol  PDI={Pb:.2f}")

print(f"\n  Training done in {time.time()-t0_train:.0f} s")

# =============================================================================
# G.  GREEDY EXPLOITATION
# =============================================================================
print("\n--- Greedy exploitation (30 random starts × 12 steps) ---")
for _ in range(30):
    oc = random_oc()
    r_oc = oc_to_real(oc)
    X, Tpk, Mn, Mw, PDI, ok = run_pfr(r_oc[0], r_oc[1], r_oc[2], r_oc[3])
    s = build_state(X, Tpk, Mn, PDI, oc)
    for __ in range(12):
        a      = policy.greedy(s)
        oc_new = oc_clip(oc + a * OC_SC)
        r_oc2  = oc_to_real(oc_new)
        X2, Tpk2, Mn2, Mw2, PDI2, ok2 = run_pfr(
            r_oc2[0], r_oc2[1], r_oc2[2], r_oc2[3]
        )
        r2 = compute_reward(X2, Tpk2, Mn2, Mw2, PDI2, ok2)
        if ok2 and r2 > best_r:
            best_r      = r2
            best_oc     = oc_new.copy()
            best_result = (X2, Tpk2, Mn2, Mw2, PDI2)
        s  = build_state(X2, Tpk2, Mn2, PDI2, oc_new)
        oc = oc_new

# =============================================================================
# H.  FINAL VALIDATION  (N=100, tight tolerance)
# =============================================================================
print("\n" + "=" * 70)
print("  Final Validation — N=100, rtol=1e-4")
print("=" * 70)

rr = oc_to_real(best_oc)
T0_f, ini_f, Tc_f, U_f = rr
print(f"  Optimal OC found by RL:")
print(f"    T₀      = {T0_f-273.15:.2f} °C")
print(f"    ini₀    = {ini_f:.5f} mol/m³")
print(f"    Tc_in   = {Tc_f-273.15:.2f} °C")
print(f"    U_heat  = {U_f:.1f} W/(m²·K)")

tv = time.time()
X_v, Tpk_v, Mn_v, Mw_v, PDI_v, ok_v = run_pfr(
    T0_f, ini_f, Tc_f, U_f, N=100, rtol=1e-4, t_frac=2.5
)
print(f"\n  Validation run done in {time.time()-tv:.1f}s  (success={ok_v})")

def chk(v, lo, hi): return '✓' if lo <= v <= hi else '✗'

kbs_opt = arrh(A_kbs, Ea_kbs, T0_f)
print(f"\n  k_bs at optimal T₀ = {kbs_opt:.3e} s⁻¹"
      f"  (τ_bs = {1/kbs_opt:.0f} s vs. τ_res = {L_react/v_flow:.0f} s)")
print(f"\n  ─── Results vs. Single-Zone LDPE Targets ────────────────────")
print(f"  Conversion X (zone) : {X_v*100:6.2f} %       [5–15 %]      {chk(X_v*100,5,15)}")
print(f"  Peak temperature    : {Tpk_v-273.15:6.1f} °C     [200–300 °C]  {chk(Tpk_v-273.15,200,300)}")
print(f"  Mn                  : {Mn_v/1e3:6.2f} kg/mol  [30–150]      {chk(Mn_v/1e3,30,150)}")
print(f"  Mw                  : {Mw_v/1e3:6.2f} kg/mol  [100–500]     {chk(Mw_v/1e3,100,500)}")
print(f"  PDI (single-zone)   : {PDI_v:6.2f}          [1.5–2.5]     {chk(PDI_v,1.5,2.5)}")

# =============================================================================
# I.  PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    'LDPE PFR — RL Optimisation v05\n'
    '(warm start + A_ktrm=1550 + PDI fixed + β-scission)',
    fontsize=11, fontweight='bold'
)

# 1. Learning curve
ax = axes[0, 0]
ax.plot(ep_rewards, alpha=0.35, color='steelblue', lw=1, label='Episode reward')
w = 10
if len(ep_rewards) >= w:
    mv = np.convolve(ep_rewards, np.ones(w)/w, mode='valid')
    ax.plot(range(w-1, len(ep_rewards)), mv, 'r', lw=2, label=f'{w}-ep moving avg')
ax.plot(ep_best_log, 'g--', lw=1.5, label='Best reward so far')
first_reset = True
for ep_idx, was_reset in enumerate(ep_was_reset):
    if was_reset:
        lbl = 'Re-randomise' if first_reset else ''
        ax.axvline(ep_idx, color='orange', ls=':', lw=1.0, alpha=0.8, label=lbl)
        first_reset = False
ax.set_xlabel('Episode'); ax.set_ylabel('Total reward')
ax.set_title('Learning Curve  (orange: re-randomise)')
ax.legend(fontsize=7)

# 2. Best results (normalised) vs targets
ax = axes[0, 1]
labels  = ['X\n(÷35%)', 'T_pk\n(÷300°C)', 'Mn\n(÷50k)', 'Mw\n(÷300k)', 'PDI\n(÷2.5)']
tgt_mid = [0.10/0.35, 250./300., 90./50., 300./300., 2.0/2.5]
if best_result:
    Xb, Tpb, Mnb, Mwb, Pb = best_result
    vals_f = [Xb/0.35, (Tpb-273.15)/300., Mnb/5e4, Mwb/3e5, Pb/2.5]
    vals_v = [X_v/0.35, (Tpk_v-273.15)/300., Mn_v/5e4, Mw_v/3e5, PDI_v/2.5]
    x  = np.arange(len(labels)); w2 = 0.3
    ax.bar(x-w2/2, vals_f, w2, alpha=0.7, label='Fast N=20',   color='steelblue')
    ax.bar(x+w2/2, vals_v, w2, alpha=0.7, label='Valid N=100', color='coral')
    ax.plot(x, tgt_mid, 'k^', ms=8, zorder=5, label='Target centre')
    ax.axhline(1., color='gray', ls='--', alpha=0.4, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.6)
    ax.set_title('Results vs. Targets (normalised)')
    ax.legend(fontsize=7)

# 3. Reward distribution
ax = axes[1, 0]
ax.hist(ep_rewards, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(np.mean(ep_rewards), color='r', ls='--', lw=1.5,
           label=f'Mean = {np.mean(ep_rewards):.1f}')
ax.axvline(best_r, color='g', ls='--', lw=1.5, label=f'Best = {best_r:.1f}')
ax.set_xlabel('Episode total reward'); ax.set_ylabel('Count')
ax.set_title('Reward Distribution'); ax.legend(fontsize=8)

# 4. Best OC bar chart + k_bs annotation
ax = axes[1, 1]
bar_labels = ['T₀ (°C)', 'ini₀×1000\n(mol/m³)', 'Tc (°C)', 'U/10\n(W/m²/K)']
bar_vals   = [T0_f-273.15, ini_f*1000., Tc_f-273.15, U_f/10.]
bars = ax.bar(bar_labels, bar_vals,
              color=['tomato', 'steelblue', 'mediumseagreen', 'goldenrod'],
              edgecolor='k', linewidth=0.8)
ax.set_title(f'Best OC  (k_bs@T₀ = {kbs_opt:.2e} s⁻¹)')
ax.set_ylabel('Value (scaled for display)')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('rl_optimization_v05_results.png', dpi=150)
print("\n  Plot saved → rl_optimization_v05_results.png")
