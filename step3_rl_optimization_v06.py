#!/usr/bin/env python3
"""
step3_rl_optimization_v06.py
LDPE Tubular PFR — RL Operating Condition Optimisation  v06

Changes from v05 (step3_rl_optimization_v05.py):
  v06-1  CTA chain-transfer added to the simulator (synced with
         step2_rxn_w_T_effect_v05.py).

         Mechanism (QSSA on CTA radical; Pₙ + CTA → Dₙ + P₁):
           dλ₁/dt|CTA = ktr_cta·[CTA]·(λ₀ − λ₁)   (reduces live chain length)
           dλ₂/dt|CTA = ktr_cta·[CTA]·(λ₀ − λ₂)   (reduces second moment)
           dμ₀/dt|CTA = +ktr_cta·[CTA]·λ₀
           dμ₁/dt|CTA = +ktr_cta·[CTA]·λ₁
           dμ₂/dt|CTA = +ktr_cta·[CTA]·λ₂
           d[CTA]/dt   = −v·d[CTA]/dz − ktr_cta·[CTA]·λ₀
           BC: [CTA](z=0) = cta_0   (proportional to F_CTA)

         Parameters: A_ktr_cta=1.0e5, Ea_ktr_cta=32 000 J/mol
         → Cs = ktr_cta/kp ≈ 0.8 at 195 °C (mercaptan-type CTA)
         → cta_0 = 10 mol/m³ reduces Mn by ~50 % vs. no CTA at T₀=195 °C

  v06-2  State expanded to 9-D (added normalised cta_0):
           [X, T_peak*, Mn*, PDI*, T0*, log_ini*, Tc*, U*, cta_0*]

  v06-3  Action expanded to 5-D (added δcta₀):
           [δT₀, δlog(ini₀), δTc, δU, δcta₀]   tanh-squashed
           δcta₀ step scale = 5 mol/m³

  v06-4  OC space extended to 5-D:
           cta_0 ∈ [0, 50] mol/m³

  v06-5  Reward Mw target updated to [50–300 kg/mol] to reflect that CTA
         reduces both Mn and Mw; the Mn target [20–100 kg/mol] is the
         primary objective (CTA is the dedicated Mn lever).

Inherited from v05:
  v05-1  β-scission (A_kbs=1e11, Ea_kbs=130 kJ/mol)

Algorithm  : REINFORCE + moving-average baseline + Adam
Policy     : 2-hidden-layer MLP (pure NumPy)
Environment: Fast LDPE PFR (N=20, Radau, loose tolerance), NV=11

State  (9-D): [X, T_peak*, Mn*, PDI*, T0*, log_ini*, Tc*, U*, cta_0*]
Action (5-D): [δT₀, δlog(ini₀), δTc, δU, δcta₀]  tanh-squashed

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
NV      = 11    # v06-1: [λ₀,λ₁,λ₂,μ₀,μ₁,μ₂,ini,mono,T,Tc,CTA]

# Arrhenius — synced with step2_rxn_w_T_effect_v05.py
A_kd      = 3.15e15;  Ea_kd      = 155_000.0
A_kp      = 6.58e4;   Ea_kp      =  29_500.0
A_ktc     = 2.0e5;    Ea_ktc     =   5_000.0
A_ktd     = 2.0e5;    Ea_ktd     =   5_000.0
A_ktrm    = 1550.;    Ea_ktrm    =  47_000.0
A_ktrp    = 3.0e-1;   Ea_ktrp    =  50_000.0
A_kbs     = 1.0e11;   Ea_kbs     = 130_000.0   # β-scission (v05-1)
A_ktr_cta = 1.0e5;    Ea_ktr_cta =  32_000.0   # CTA chain transfer (v06-1)

def arrh(A, Ea, T):
    return A * np.exp(-Ea / (R_gas * T))

def _build_jac(N):
    rows, cols = [], []
    for i in range(N):
        for j in range(NV):
            r = i * NV + j
            for k in range(NV):
                rows.append(r); cols.append(i * NV + k)
            # upwind: forward-flowing species (indices 0-8 and 10=CTA)
            if (j < 9 or j == 10) and i > 0:
                rows.append(r); cols.append((i-1)*NV + j)
            # downwind: counter-current coolant (index 9)
            if j == 9 and i < N - 1:
                rows.append(r); cols.append((i+1)*NV + 9)
    return csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(N*NV, N*NV)
    )

# =============================================================================
# B.  FAST LDPE PFR SIMULATOR  (v06-1: NV=11, CTA added)
# =============================================================================
def run_pfr(T_0, ini_0, Tc_in, U_heat, cta_0=0., N=20, rtol=5e-3, t_frac=1.8):
    """
    Integrate LDPE PFR moment ODEs to pseudo-steady state.
    cta_0 : inlet CTA concentration [mol/m³] (set by F_CTA).
    Returns (X_exit, T_peak_K, Mn_g/mol, Mw_g/mol, PDI, success).
    """
    h_r = 4.0 * U_heat / (rho * Cp * D)
    h_j = U_heat * np.pi * D / (rho_c * Cp_c * A_c)
    dz  = L_react / (N - 1)
    tau = L_react / v_flow
    jac = _build_jac(N)
    atol = np.tile([1e-12, 1e-9, 1e-5, 1e-10, 1e-7, 1e-2, 1e-7, 1., 1e-3, 1e-3, 1e-5], N)

    def odes(t, y):
        s = y.reshape(N, NV)
        l0, l1, l2 = s[:,0], s[:,1], s[:,2]
        m0, m1, m2 = s[:,3], s[:,4], s[:,5]
        ini, mono, T, Tc = s[:,6], s[:,7], s[:,8], s[:,9]
        cta = s[:,10]   # v06-1: [CTA] state variable

        kd      = arrh(A_kd,      Ea_kd,      T)
        kp      = arrh(A_kp,      Ea_kp,      T)
        ktc     = arrh(A_ktc,     Ea_ktc,     T)
        ktd     = arrh(A_ktd,     Ea_ktd,     T)
        ktrm    = arrh(A_ktrm,    Ea_ktrm,    T)
        ktrp    = arrh(A_ktrp,    Ea_ktrp,    T)
        kbs     = arrh(A_kbs,     Ea_kbs,     T)
        ktr_cta = arrh(A_ktr_cta, Ea_ktr_cta, T)   # v06-1

        eps  = 1e-12
        mu3h = np.where((m0 > eps) & (m1 > eps),
                        m2*(2*m0*m2 - m1**2) / (m0*m1 + eps), 0.)
        mu3  = np.maximum(mu3h, np.where(m1 > eps, m2**2 / (m1 + eps), 0.))

        # β-scission: Pₙ → P_{n/2} + D_{n/2}
        # v06-1 CTA: Pₙ + CTA → Dₙ + P₁
        Rl0 = 2*f_eff*kd*ini - (ktc+ktd)*l0**2
        Rl1 = (kp*mono*l0 + ktrm*mono*(l0-l1)
               + ktrp*(l0*m2-l1*m1) - (ktc+ktd)*l0*l1
               - kbs*l1/2.
               + ktr_cta*cta*(l0 - l1))              # v06-1
        Rl2 = (kp*mono*(2*l1+l0) + ktrm*mono*(l0-l2)
               + ktrp*(l0*mu3-l2*m1) - (ktc+ktd)*l0*l2
               - kbs*l2*2./3.
               + ktr_cta*cta*(l0 - l2))              # v06-1
        Rm0 = (ktrm*mono*l0 + (0.5*ktc+ktd)*l0**2
               + kbs*l0
               + ktr_cta*cta*l0)                     # v06-1
        Rm1 = (ktrm*mono*l1 + ktrp*(l1*m1-l0*m2) + (ktc+ktd)*l0*l1
               + kbs*l1/2.
               + ktr_cta*cta*l1)                     # v06-1
        Rm2 = (ktrm*mono*l2 + ktd*l0*l2
               + ktc*(l0*l2+l1**2) + ktrp*(l2*m1-l0*mu3)
               + kbs*l2/3.
               + ktr_cta*cta*l2)                     # v06-1
        Ri  = -kd*ini
        Rm  = -kp*mono*l0
        RT  = (-dH_p)/(rho*Cp)*kp*mono*l0 - h_r*(T - Tc)
        T_safe = 320. + 273.15
        RT -= np.maximum(0., (T - T_safe)) * 20.0
        RTc = h_j*(T - Tc)
        # v06-1: CTA mass balance — consumed by chain transfer
        Rcta = -ktr_cta * cta * l0

        dy = np.zeros_like(s)
        # Forward-flowing variables (upwind, indices 0-8)
        for j, (C, R, BC) in enumerate(zip(
            [l0, l1, l2, m0, m1, m2, ini, mono, T],
            [Rl0,Rl1,Rl2,Rm0,Rm1,Rm2, Ri,  Rm,  RT],
            [0., 0., 0., 0., 0., 0., ini_0, mono_0, T_0]
        )):
            Cu = np.empty(N); Cu[0] = BC; Cu[1:] = C[:-1]
            dy[0, j] = 0.
            dy[1:, j] = R[1:] - v_flow*(C[1:] - Cu[1:])/dz

        # Counter-current coolant (index 9)
        Cd = np.empty(N); Cd[:-1] = Tc[1:]; Cd[-1] = Tc_in
        dy[:N-1, 9] = RTc[:N-1] + v_cool*(Cd[:N-1] - Tc[:N-1])/dz
        dy[N-1,  9] = 0.

        # v06-1: CTA (upwind, index 10)
        Cu_cta = np.empty(N); Cu_cta[0] = cta_0; Cu_cta[1:] = cta[:-1]
        dy[0,  10] = 0.
        dy[1:, 10] = Rcta[1:] - v_flow*(cta[1:] - Cu_cta[1:])/dz

        return dy.ravel()

    y0 = np.zeros((N, NV))
    y0[:, 7]  = mono_0
    y0[:, 8]  = T_0
    y0[:, 9]  = Tc_in
    y0[:, 10] = cta_0      # v06-1: initialise CTA profile
    y0[0,  6] = ini_0
    y0[N-1, 9] = Tc_in

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
# C.  REWARD  (v06-5: Mw target [50–300], Mn target [20–100])
# =============================================================================
def compute_reward(X, Tpk, Mn, Mw, PDI, ok):
    """
    Multi-objective reward for LDPE single-injection-zone with CTA control.
    v06-5: Mw target [50–300 kg/mol]  (CTA reduces Mw together with Mn).
           Mn target [20–100 kg/mol]  (primary CTA control objective).
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

    # --- Mn  [20–100 kg/mol]  primary CTA objective ---
    if 20. <= Mn_k <= 100.:
        r += 10.
    else:
        r -= 6. * max(20. - Mn_k, Mn_k - 100.) / 40.

    # --- Mw  [50–300 kg/mol]  (v06-5: adjusted for CTA-enabled system) ---
    if 50. <= Mw_k <= 300.:
        r += 6.
    else:
        r -= 3. * max(50. - Mw_k, Mw_k - 300.) / 100.

    # --- PDI  (single-zone MOM target [1.5–2.5]) ---
    if 1.5 <= PDI <= 2.5:
        r += 5.
    elif PDI < 1.5:
        r -= 3.
    elif PDI > 4.0:
        r -= 2. * (PDI - 4.0) / 4.0

    return float(r)

# =============================================================================
# D.  OPERATING CONDITION SPACE  (v06-3/4: 5-D; added cta_0)
# =============================================================================
# OC vector: [T₀(K), log(ini₀), Tc_in(K), U_heat(W/m²K), cta_0(mol/m³)]
OC_LO = np.array([413.15, np.log(0.003), 363.15,  400.,  0.])
OC_HI = np.array([483.15, np.log(0.100), 443.15, 3000., 50.])
OC_SC = np.array([8.,     0.30,           8.,    250.,   5.])  # action step scales

def oc_clip(oc):    return np.clip(oc, OC_LO, OC_HI)
def oc_to_real(oc): r = oc.copy(); r[1] = np.exp(oc[1]); return r
def oc_norm(oc):    return (oc - OC_LO) / (OC_HI - OC_LO) * 2. - 1.
def random_oc():    return OC_LO + np.random.rand(5) * (OC_HI - OC_LO)

def build_state(X, Tpk, Mn, PDI, oc):
    """9-D state vector (v06-2)."""
    s = np.array([
        X / 0.40,
        (Tpk - 273.15) / 400.,
        Mn  / 6e4,
        PDI / 25.,
        *oc_norm(oc)           # 5 normalised OC components
    ], dtype=np.float64)
    return np.clip(s, -3., 3.)

# =============================================================================
# E.  GAUSSIAN MLP POLICY  (v06-2/3: s_dim=9, a_dim=5)
# =============================================================================
class GaussianMLP:
    def __init__(self, s_dim=9, h_dim=64, a_dim=5, lr=4e-3, seed=7):
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
#     warm start: each episode begins from previous episode's final OC.
#     Re-randomisation every N_RESET = 25 episodes.
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

print("=" * 72)
print("  LDPE PFR — RL Optimisation  v06  (REINFORCE + CTA control)")
print("=" * 72)
print(f"  v05-1  β-scission:       A_kbs=1e11, Ea_kbs=130 kJ/mol")
print(f"  v06-1  CTA transfer:     A_ktr_cta=1e5, Ea_ktr_cta=32 kJ/mol  Cs≈0.8")
print(f"  v06-2  State dim: 9  (added cta_0*)")
print(f"  v06-3  Action dim: 5  (added δcta₀,  step scale={OC_SC[4]} mol/m³)")
print(f"  v06-4  OC space: cta_0 ∈ [{OC_LO[4]:.0f}, {OC_HI[4]:.0f}] mol/m³")
print(f"  v06-5  Reward: Mn [20–100], Mw [50–300] kg/mol")
print(f"  Episodes : {N_EPS} × {N_STEPS} steps = {N_EPS*N_STEPS} simulations (N=20)")
rr_lo = oc_to_real(OC_LO); rr_hi = oc_to_real(OC_HI)
print(f"  OC search space:")
print(f"    T₀      : {rr_lo[0]-273.15:.0f} – {rr_hi[0]-273.15:.0f} °C")
print(f"    ini₀    : {rr_lo[1]:.4f} – {rr_hi[1]:.3f} mol/m³")
print(f"    Tc_in   : {rr_lo[2]-273.15:.0f} – {rr_hi[2]-273.15:.0f} °C")
print(f"    U_heat  : {rr_lo[3]:.0f} – {rr_hi[3]:.0f} W/(m²·K)")
print(f"    cta_0   : {rr_lo[4]:.0f} – {rr_hi[4]:.0f} mol/m³")
print("=" * 72)
t0_train = time.time()

oc = random_oc()

for ep in range(N_EPS):
    rerandom = (ep > 0 and ep % N_RESET == 0)
    if rerandom:
        oc = random_oc()
    ep_was_reset.append(rerandom)

    r_oc = oc_to_real(oc)
    X, Tpk, Mn, Mw, PDI, ok = run_pfr(r_oc[0], r_oc[1], r_oc[2], r_oc[3], r_oc[4])
    s = build_state(X, Tpk, Mn, PDI, oc)

    traj    = []
    ep_rtot = 0.

    for step in range(N_STEPS):
        a, u, lp = policy.sample(s)
        oc_new   = oc_clip(oc + a * OC_SC)
        r_oc2    = oc_to_real(oc_new)
        X2, Tpk2, Mn2, Mw2, PDI2, ok2 = run_pfr(
            r_oc2[0], r_oc2[1], r_oc2[2], r_oc2[3], r_oc2[4]
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
                  f"Tc={rr[2]-273.15:.1f}°C  U={rr[3]:.0f}  cta₀={rr[4]:.2f} mol/m³")
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
    X, Tpk, Mn, Mw, PDI, ok = run_pfr(r_oc[0], r_oc[1], r_oc[2], r_oc[3], r_oc[4])
    s = build_state(X, Tpk, Mn, PDI, oc)
    for __ in range(12):
        a      = policy.greedy(s)
        oc_new = oc_clip(oc + a * OC_SC)
        r_oc2  = oc_to_real(oc_new)
        X2, Tpk2, Mn2, Mw2, PDI2, ok2 = run_pfr(
            r_oc2[0], r_oc2[1], r_oc2[2], r_oc2[3], r_oc2[4]
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
print("\n" + "=" * 72)
print("  Final Validation — N=100, rtol=1e-4")
print("=" * 72)

rr = oc_to_real(best_oc)
T0_f, ini_f, Tc_f, U_f, cta_f_oc = rr
print(f"  Optimal OC found by RL:")
print(f"    T₀      = {T0_f-273.15:.2f} °C")
print(f"    ini₀    = {ini_f:.5f} mol/m³")
print(f"    Tc_in   = {Tc_f-273.15:.2f} °C")
print(f"    U_heat  = {U_f:.1f} W/(m²·K)")
print(f"    cta_0   = {cta_f_oc:.3f} mol/m³  (F_CTA lever)")

tv = time.time()
X_v, Tpk_v, Mn_v, Mw_v, PDI_v, ok_v = run_pfr(
    T0_f, ini_f, Tc_f, U_f, cta_f_oc, N=100, rtol=1e-4, t_frac=2.5
)
print(f"\n  Validation run done in {time.time()-tv:.1f}s  (success={ok_v})")

def chk(v, lo, hi): return '✓' if lo <= v <= hi else '✗'

kbs_opt = arrh(A_kbs, Ea_kbs, T0_f)
kct_opt = arrh(A_ktr_cta, Ea_ktr_cta, T0_f)
Cs_opt  = kct_opt / arrh(A_kp, Ea_kp, T0_f)
print(f"\n  k_bs at optimal T₀ = {kbs_opt:.3e} s⁻¹")
print(f"  ktr_cta at opt T₀  = {kct_opt:.3e} m³/(mol·s)  Cs={Cs_opt:.3f}")
print(f"\n  ─── Results vs. CTA-Enabled LDPE Targets ────────────────────────")
print(f"  Conversion X (zone) : {X_v*100:6.2f} %       [5–15 %]      {chk(X_v*100,5,15)}")
print(f"  Peak temperature    : {Tpk_v-273.15:6.1f} °C     [200–300 °C]  {chk(Tpk_v-273.15,200,300)}")
print(f"  Mn                  : {Mn_v/1e3:6.2f} kg/mol  [20–100]      {chk(Mn_v/1e3,20,100)}")
print(f"  Mw                  : {Mw_v/1e3:6.2f} kg/mol  [50–300]      {chk(Mw_v/1e3,50,300)}")
print(f"  PDI (single-zone)   : {PDI_v:6.2f}          [1.5–2.5]     {chk(PDI_v,1.5,2.5)}")

# =============================================================================
# I.  PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    'LDPE PFR — RL Optimisation v06\n'
    '(5-D actions: T₀, ini₀, Tc, U, cta₀  |  CTA = primary Mn lever)',
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

# 2. Best results vs targets
ax = axes[0, 1]
labels  = ['X\n(÷35%)', 'T_pk\n(÷300°C)', 'Mn\n(÷50k)', 'Mw\n(÷300k)', 'PDI\n(÷2.5)']
tgt_mid = [0.10/0.35, 250./300., 60./50., 175./300., 2.0/2.5]
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
ax = axes[0, 2]
ax.hist(ep_rewards, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(np.mean(ep_rewards), color='r', ls='--', lw=1.5,
           label=f'Mean = {np.mean(ep_rewards):.1f}')
ax.axvline(best_r, color='g', ls='--', lw=1.5, label=f'Best = {best_r:.1f}')
ax.set_xlabel('Episode total reward'); ax.set_ylabel('Count')
ax.set_title('Reward Distribution'); ax.legend(fontsize=8)

# 4. Best OC bar chart (all 5 OCs)
ax = axes[1, 0]
bar_labels = ['T₀ (°C)', 'ini₀×1000\n(mol/m³)', 'Tc (°C)', 'U/10\n(W/m²/K)',
              'cta₀\n(mol/m³)']
bar_vals   = [T0_f-273.15, ini_f*1000., Tc_f-273.15, U_f/10., cta_f_oc]
bar_colors = ['tomato', 'steelblue', 'mediumseagreen', 'goldenrod', 'mediumpurple']
ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor='k', linewidth=0.8)
ax.set_title(f'Best OC  (5-D: T₀, ini₀, Tc, U, cta₀)')
ax.set_ylabel('Value (scaled for display)')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 5. CTA action distribution from training
ax = axes[1, 1]
# Collect final cta_0 values from each episode's last action
all_cta = []
if best_oc is not None:
    rr_all = oc_to_real(OC_LO + np.random.rand(200, 5) * (OC_HI - OC_LO))
    all_cta = rr_all[:, 4]
    ax.hist(all_cta, bins=20, color='mediumpurple', edgecolor='white', alpha=0.8)
    ax.axvline(cta_f_oc, color='r', ls='--', lw=2, label=f'Best cta₀={cta_f_oc:.1f}')
    ax.set_xlabel('cta_0 (mol/m³)'); ax.set_ylabel('Count (random OC sample)')
    ax.set_title('CTA₀ search space distribution')
    ax.legend(fontsize=8)

# 6. Mn vs cta_0 — sanity check of CTA lever (fast sims)
ax = axes[1, 2]
cta_scan = np.linspace(0., 40., 12)
mn_scan  = []
for c0 in cta_scan:
    Xi, Tpi, Mni, Mwi, PDIi, oki = run_pfr(T0_f, ini_f, Tc_f, U_f, c0, N=20)
    mn_scan.append(Mni/1e3 if oki else np.nan)
ax.plot(cta_scan, mn_scan, 'o-', lw=2, color='mediumpurple', ms=5)
ax.axvline(cta_f_oc, color='r', ls='--', lw=1.5, label=f'Optimal cta₀={cta_f_oc:.1f}')
ax.axhline(20.,  color='gray', ls=':', lw=0.8, label='Mn bounds [20,100]')
ax.axhline(100., color='gray', ls=':', lw=0.8)
ax.set_xlabel('cta_0 (mol/m³)')
ax.set_ylabel('Mn (kg/mol)')
ax.set_title('Mn vs. cta₀ at optimal T₀, ini₀, Tc, U\n(CTA as Mn control lever)')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('rl_optimization_v06_results.png', dpi=150)
print("\n  Plot saved → rl_optimization_v06_results.png")
