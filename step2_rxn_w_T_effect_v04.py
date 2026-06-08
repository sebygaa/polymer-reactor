# %%
# Importing Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import time

# =============================================================================
# LDPE Free-Radical Polymerisation — Dynamic 1D PFR Model  v04
#
# Key changes from v03:
#   v04-1  β-scission added (two new Arrhenius constants: A_kbs, Ea_kbs).
#          At T > 250 °C, mid-chain radicals formed by chain transfer to polymer
#          (ktrp) can cleave the backbone via β-scission, shortening both the
#          live radical and producing a dead chain fragment.  In this MOM model
#          all living radicals are treated as potentially susceptible (lumped
#          effective treatment; no separate mid-chain radical species).
#
#          Mechanism (symmetric random-scission approximation):
#            Pₙ  →  P_{n/2}  +  D_{n/2}          rate = k_bs · λ₀
#
#          Moment contributions (derived from population-balance integration):
#            dλ₁/dt|_bs = −k_bs · λ₁/2          live radicals shorten
#            dλ₂/dt|_bs = −k_bs · λ₂ · 2/3
#            dμ₀/dt|_bs = +k_bs · λ₀             one dead fragment per event
#            dμ₁/dt|_bs = +k_bs · λ₁/2
#            dμ₂/dt|_bs = +k_bs · λ₂/3
#
#          Parameters (effective lumped values for MOM without explicit
#          mid-chain radical tracking; see physical basis section below):
#            A_kbs  = 1.0e11  s⁻¹
#            Ea_kbs = 130 000  J/mol
#
#          Temperature dependence of k_bs:
#            195 °C : k_bs ≈ 3.8e-4 s⁻¹  → negligible (< 1 % Mn change)
#            250 °C : k_bs ≈ 1.1e-2 s⁻¹  → minor  (~  4 % Mn change at hot spot)
#            280 °C : k_bs ≈ 4.9e-2 s⁻¹  → notable (~10 % Mn change at hot spot)
#            300 °C : k_bs ≈ 1.3e-1 s⁻¹  → significant; Mn decreases visibly
#
#   v04-2  β-scission effect study added: base case and T-sensitivity runs
#          are now compared with β-scission disabled (k_bs = 0) to isolate
#          the contribution at each inlet temperature.
# =============================================================================

# %%
# ---- Physical / Process Constants ------------------------------------------
R_gas   = 8.3145
Mw_mono = 28.054
f_eff   = 0.8

L    = 500.0
D    = 0.05
D_j  = 0.07
A_r  = np.pi / 4 * D**2
A_c  = np.pi / 4 * (D_j**2 - D**2)
v    = 12.0
v_c  = 1.0
rho  = 600_000.0
Cp   = 1.7
dH_p = -93_000.0

U_heat  = 1500.0
Tc_in   = 130.0 + 273.15
rho_c   = 900_000.0
Cp_c    = 4.18

h_r = 4.0 * U_heat / (rho * Cp * D)
h_j = U_heat * np.pi * D / (rho_c * Cp_c * A_c)

P_ref_bar = 2000.0
DV_kp     = -27e-6    # activation volume for kp [m³/mol]  (Buback 2000)

# %%
# ---- Arrhenius Parameters ---------------------------------------------------
A_kd   = 3.15e15;  Ea_kd   = 155_000.0   # initiator decomp
A_kp   = 6.58e4;   Ea_kp   =  29_500.0   # propagation
A_ktc  = 2.0e5;    Ea_ktc  =   5_000.0   # termination by combination
A_ktd  = 2.0e5;    Ea_ktd  =   5_000.0   # termination by disproportionation
A_ktrm = 1550.;    Ea_ktrm =  47_000.0   # effective CTA + monomer transfer
A_ktrp = 3.0e-1;   Ea_ktrp =  50_000.0   # chain transfer to polymer (LCB)
# v04-1: β-scission of living radicals (lumped effective, no MCR species)
A_kbs  = 1.0e11;   Ea_kbs  = 130_000.0   # β-scission

def arrhenius(A, Ea, T):
    return A * np.exp(-Ea / (R_gas * T))

# %%
# ---- Spatial Grid ----------------------------------------------------------
N  = 200
NV = 10
z  = np.linspace(0.0, L, N)
dz = z[1] - z[0]
print(f'Grid: N={N}, dz={dz:.2f} m, τ_res={L/v:.1f} s')

# %%
# ---- Jacobian Sparsity -----------------------------------------------------
def build_jac_sparsity(N, NV=10):
    rows, cols = [], []
    for i in range(N):
        for j in range(NV):
            row = i * NV + j
            for k in range(NV):
                rows.append(row); cols.append(i * NV + k)
            if j < 9 and i > 0:
                rows.append(row); cols.append((i - 1) * NV + j)
            if j == 9 and i < N - 1:
                rows.append(row); cols.append((i + 1) * NV + 9)
    n = N * NV
    return csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))

jac_sp = build_jac_sparsity(N, NV)
print(f'Jacobian: {N*NV}×{N*NV}, nnz={jac_sp.nnz}')

# %%
# ---- Base Feed Conditions --------------------------------------------------
mono_0 = 2.005e4
ini_0  = 0.01
T_0    = 195.0 + 273.15

# %%
# ---- Tolerance Array -------------------------------------------------------
atol_per_var = np.array([1e-14, 1e-11, 1e-7, 1e-11, 1e-8, 1e-3, 1e-8, 1e-1, 1e-4, 1e-4])
atol_vec     = np.tile(atol_per_var, N)

# %%
# ---- ODE Builder -----------------------------------------------------------
def make_odes(ini_0_, mono_0_, T_0_, Tc_in_, h_r_, h_j_,
              kp_factor=1.0, N_=None, enable_bs=True):
    """
    Return ODE function for given operating conditions.
    enable_bs : include β-scission terms (v04-1). Set False to replicate v03.
    """
    N_  = N_ or N
    dz_ = L / (N_ - 1)
    T_safe = 300. + 273.15

    def pfr_odes(t, y):
        s = y.reshape(N_, NV)
        lamb0=s[:,0]; lamb1=s[:,1]; lamb2=s[:,2]
        mu0  =s[:,3]; mu1  =s[:,4]; mu2  =s[:,5]
        ini  =s[:,6]; mono =s[:,7]; T=s[:,8]; Tc=s[:,9]

        kd   = arrhenius(A_kd,   Ea_kd,   T)
        kp   = arrhenius(A_kp,   Ea_kp,   T) * kp_factor
        ktc  = arrhenius(A_ktc,  Ea_ktc,  T)
        ktd  = arrhenius(A_ktd,  Ea_ktd,  T)
        ktrm = arrhenius(A_ktrm, Ea_ktrm, T)
        ktrp = arrhenius(A_ktrp, Ea_ktrp, T)
        kbs  = arrhenius(A_kbs,  Ea_kbs,  T) if enable_bs else np.zeros_like(T)

        eps3  = 1e-12
        valid = (mu0 > eps3) & (mu1 > eps3)
        mu3_hk = np.where(valid,
                          mu2*(2.*mu0*mu2-mu1**2)/(mu0*mu1+eps3), 0.)
        mu3_cs = np.where(mu1 > eps3, mu2**2/(mu1+eps3), 0.)
        mu3    = np.maximum(mu3_hk, mu3_cs)

        # ---- Living chain moments -------------------------------------------
        # β-scission: Pₙ → P_{n/2} + D_{n/2}   (symmetric scission, rate = k_bs·λ₀)
        #   dλ₀|_bs = 0      (radical count conserved: one consumed, one created)
        #   dλ₁|_bs = −k_bs·λ₁/2    (new radical is half the length)
        #   dλ₂|_bs = −k_bs·λ₂·2/3  (second moment of new radical, uniform split)
        R_l0 = 2.*f_eff*kd*ini - (ktc+ktd)*lamb0**2
        R_l1 = (kp*mono*lamb0 + ktrm*mono*(lamb0-lamb1)
                + ktrp*(lamb0*mu2-lamb1*mu1) - (ktc+ktd)*lamb0*lamb1
                - kbs*lamb1/2.)
        R_l2 = (kp*mono*(2.*lamb1+lamb0) + ktrm*mono*(lamb0-lamb2)
                + ktrp*(lamb0*mu3-lamb2*mu1) - (ktc+ktd)*lamb0*lamb2
                - kbs*lamb2*2./3.)

        # ---- Dead chain moments ---------------------------------------------
        # β-scission dead fragment D_{n/2} contributes to μ moments:
        #   dμ₀|_bs = +k_bs·λ₀      one new dead chain per scission event
        #   dμ₁|_bs = +k_bs·λ₁/2   dead fragment length = half of parent
        #   dμ₂|_bs = +k_bs·λ₂/3   second moment of dead fragment (uniform split)
        R_m0 = (ktrm*mono*lamb0 + (0.5*ktc+ktd)*lamb0**2
                + kbs*lamb0)
        R_m1 = (ktrm*mono*lamb1 + ktrp*(lamb1*mu1-lamb0*mu2)
                + (ktc+ktd)*lamb0*lamb1
                + kbs*lamb1/2.)
        R_m2 = (ktrm*mono*lamb2 + ktd*lamb0*lamb2
                + ktc*(lamb0*lamb2+lamb1**2) + ktrp*(lamb2*mu1-lamb0*mu3)
                + kbs*lamb2/3.)

        R_ini  = -kd * ini
        R_mono = -kp * mono * lamb0
        RT     = (-dH_p)/(rho*Cp)*kp*mono*lamb0 - h_r_*(T - Tc)
        RT    -= np.maximum(0., T - T_safe) * 10.0
        R_Tc   = h_j_ * (T - Tc)

        dydt = np.zeros_like(s)
        for j, (C, R, BC) in enumerate(zip(
            [lamb0,lamb1,lamb2, mu0,mu1,mu2,   ini,   mono, T],
            [R_l0, R_l1, R_l2, R_m0,R_m1,R_m2, R_ini, R_mono, RT],
            [0.,   0.,   0.,   0.,  0.,  0.,    ini_0_, mono_0_, T_0_]
        )):
            C_up     = np.empty(N_)
            C_up[0]  = BC
            C_up[1:] = C[:-1]
            dCdz     = (C - C_up) / dz_
            dydt[0, j]  = 0.
            dydt[1:, j] = R[1:] - v * dCdz[1:]

        Tc_dn      = np.empty(N_)
        Tc_dn[:-1] = Tc[1:]
        Tc_dn[-1]  = Tc_in_
        dTcdz      = (Tc_dn - Tc) / dz_
        dydt[:N_-1, 9] = R_Tc[:N_-1] + v_c * dTcdz[:N_-1]
        dydt[N_-1,  9] = 0.
        return dydt.ravel()

    return pfr_odes

# %%
# ---- Run one simulation ----------------------------------------------------
def run_pfr(T0_C=195., ini_0_=0.01, Tc_C=130., U_=1500., P_bar=2000.,
            N_=200, rtol_=1e-4, t_frac=4., enable_bs=True):
    T0_K    = T0_C + 273.15
    Tc_K    = Tc_C + 273.15
    h_r_    = 4.*U_/(rho*Cp*D)
    h_j_    = U_*np.pi*D/(rho_c*Cp_c*A_c)
    kp_fac  = np.exp(-DV_kp*(P_bar-P_ref_bar)*1e5/(R_gas*T0_K))
    jac_    = build_jac_sparsity(N_, NV)
    atol_   = np.tile(atol_per_var, N_)
    tau     = L / v
    t_end   = t_frac * tau

    y0 = np.zeros((N_, NV))
    y0[:, 7] = mono_0;  y0[:, 8] = T0_K;  y0[:, 9] = Tc_K
    y0[0,  6] = ini_0_; y0[N_-1, 9] = Tc_K

    odes = make_odes(ini_0_, mono_0, T0_K, Tc_K, h_r_, h_j_,
                     kp_fac, N_, enable_bs=enable_bs)

    t0 = time.time()
    sol = solve_ivp(odes, (0., t_end), y0.ravel(), method='Radau',
                    t_eval=np.linspace(0., t_end, 300),
                    rtol=rtol_, atol=atol_,
                    jac_sparsity=jac_, dense_output=False)
    elapsed = time.time() - t0

    nt = len(sol.t)
    Y  = sol.y.T.reshape(nt, N_, NV)
    mu0_f = Y[-1,:,3]; mu1_f = Y[-1,:,4]; mu2_f = Y[-1,:,5]
    mono_f= Y[-1,:,7]; T_f   = Y[-1,:,8]; Tc_f  = Y[-1,:,9]
    eps = 1e-30
    Mn_z  = Mw_mono * (mu1_f+eps) / (mu0_f+eps)
    Mw_z  = Mw_mono * (mu2_f+eps) / (mu1_f+eps)
    PDI_z = Mw_z / Mn_z
    X_z   = 1. - mono_f / mono_0

    return dict(
        ok       = sol.status == 0,
        t_cpu    = elapsed,
        Y        = Y,
        sol      = sol,
        z        = np.linspace(0., L, N_),
        X_exit   = float(X_z[-1]),
        T_peak   = float(T_f.max() - 273.15),
        Mn_exit  = float(Mn_z[-1]) / 1000.,
        Mw_exit  = float(Mw_z[-1]) / 1000.,
        PDI_exit = float(PDI_z[-1]),
        Mn_z     = Mn_z / 1000.,
        Mw_z     = Mw_z / 1000.,
        PDI_z    = PDI_z,
        X_z      = X_z,
        T_z      = T_f - 273.15,
        Tc_z     = Tc_f - 273.15,
    )

# %%
# ---- β-scission rate check at key temperatures -----------------------------
print('\n--- β-scission rate constant vs. temperature ---')
print(f'{"T (°C)":>8}  {"k_bs (s⁻¹)":>12}  {"k_bs/kp/mono":>14}  note')
print('-'*58)
for Tc_ in [195., 210., 230., 250., 270., 280., 300.]:
    T_K  = Tc_ + 273.15
    kbs_ = arrhenius(A_kbs, Ea_kbs, T_K)
    kp_  = arrhenius(A_kp,  Ea_kp,  T_K)
    ratio = kbs_ / (kp_ * mono_0)
    tau_  = L / v
    # approximate Mn change over full τ (upper bound: assumes T constant)
    dMn   = (1. - np.exp(-kbs_*tau_/2.)) * 100.
    print(f'{Tc_:8.1f}  {kbs_:12.4e}  {ratio:14.4e}  ΔMn≤{dMn:.1f}%')

# %%
# ---- Base-case run (with β-scission) ----------------------------------------
print('\n--- Base case  (T₀=195 °C, ini₀=0.01, Tc=130 °C, U=1500, β-scission ON) ---')
base = run_pfr()
print(f'  CPU time    : {base["t_cpu"]:.1f} s   solver ok={base["ok"]}')
print(f'  X (exit)    : {base["X_exit"]*100:.2f} %       [5–15 %]  {"✓" if 0.05<=base["X_exit"]<=0.15 else "✗"}')
print(f'  T peak      : {base["T_peak"]:.1f} °C    [200–300 °C]  {"✓" if 200<=base["T_peak"]<=300 else "✗"}')
print(f'  Mn (exit)   : {base["Mn_exit"]:.2f} kg/mol  [30–150]  {"✓" if 30<=base["Mn_exit"]<=150 else "✗"}')
print(f'  Mw (exit)   : {base["Mw_exit"]:.2f} kg/mol  [100–500] {"✓" if 100<=base["Mw_exit"]<=500 else "✗"}')
print(f'  PDI         : {base["PDI_exit"]:.2f}          [3–15]    {"✓" if 3<=base["PDI_exit"]<=15 else "✗"}')

# %%
# ---- Temperature sensitivity with and without β-scission -------------------
T0_cases = [180., 185., 190., 195., 200., 205., 210.]
print('\n--- Temperature sensitivity  (N=100, ini₀=0.01, Tc=130 °C, U=1500) ---')
print(f'{"T₀":>5}  {"T_pk":>6}  {"X%":>5}  '
      f'{"Mn_bs":>8} {"Mn_no":>8}  {"ΔMn%":>7}  '
      f'{"Mw_bs":>8} {"Mw_no":>8}  {"PDI_bs":>7} {"PDI_no":>7}  ok')
print('-'*96)
T_sens_bs  = {}
T_sens_nobs = {}
for T0c in T0_cases:
    r_bs  = run_pfr(T0_C=T0c, N_=100, rtol_=1e-3, t_frac=3., enable_bs=True)
    r_nbs = run_pfr(T0_C=T0c, N_=100, rtol_=1e-3, t_frac=3., enable_bs=False)
    T_sens_bs[T0c]   = r_bs
    T_sens_nobs[T0c] = r_nbs
    dMn = (r_nbs['Mn_exit'] - r_bs['Mn_exit']) / r_nbs['Mn_exit'] * 100.
    print(f'{T0c:5.1f}  {r_bs["T_peak"]:6.1f}  {r_bs["X_exit"]*100:5.2f}  '
          f'{r_bs["Mn_exit"]:8.2f} {r_nbs["Mn_exit"]:8.2f}  {dMn:7.2f}%  '
          f'{r_bs["Mw_exit"]:8.2f} {r_nbs["Mw_exit"]:8.2f}  '
          f'{r_bs["PDI_exit"]:7.2f} {r_nbs["PDI_exit"]:7.2f}  '
          f'{"ok" if r_bs["ok"] else "FAIL"}')

# %%
# ---- Figure 1: Base-case steady-state profiles (β-scission ON) -------------
fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
fig1.suptitle(f'LDPE PFR v04 — Base-case steady state  '
              f'(T₀=195 °C, β-scission ON)', fontsize=10)
zb = base['z']

ax = axes[0,0]
ax.plot(zb, base['T_z'],  'r',   lw=2, label='T reactor')
ax.plot(zb, base['Tc_z'], 'b--', lw=2, label='Tc coolant')
ax.axhline(200, color='gray', ls=':', lw=0.8)
ax.axhline(300, color='gray', ls=':', lw=0.8)
ax.set(xlabel='z (m)', ylabel='T (°C)', title='Temperature')
ax.legend(fontsize=8)

axes[0,1].plot(zb, base['X_z']*100, 'g', lw=2)
axes[0,1].axhline(5,  color='gray', ls=':', lw=0.8)
axes[0,1].axhline(15, color='gray', ls=':', lw=0.8)
axes[0,1].set(xlabel='z (m)', ylabel='Conversion (%)', title='Monomer conversion')

axes[0,2].plot(zb, base['Mn_z'], lw=2)
axes[0,2].axhline(30,  color='gray', ls=':', lw=0.8)
axes[0,2].axhline(150, color='gray', ls=':', lw=0.8)
axes[0,2].set(xlabel='z (m)', ylabel='Mn (kg/mol)', title='Mn')

axes[0,3].plot(zb, base['Mw_z'], lw=2)
axes[0,3].axhline(100, color='gray', ls=':', lw=0.8)
axes[0,3].axhline(500, color='gray', ls=':', lw=0.8)
axes[0,3].set(xlabel='z (m)', ylabel='Mw (kg/mol)', title='Mw')

axes[1,0].plot(zb, base['PDI_z'], lw=2)
axes[1,0].axhline(3,  color='gray', ls=':', lw=0.8)
axes[1,0].axhline(15, color='gray', ls=':', lw=0.8)
axes[1,0].set(xlabel='z (m)', ylabel='PDI', title='PDI')

Y   = base['Y']
t_arr = base['sol'].t
T_dyn = Y[:,:,8] - 273.15
X_dyn = 1. - Y[:,:,7] / mono_0

axes[1,1].contourf(zb, t_arr, T_dyn, levels=40, cmap='hot')
axes[1,1].set(xlabel='z (m)', ylabel='t (s)', title='T(z,t) map')

axes[1,2].contourf(zb, t_arr, X_dyn*100, levels=40, cmap='viridis')
axes[1,2].set(xlabel='z (m)', ylabel='t (s)', title='X(z,t) map')

axes[1,3].plot(zb, base['Y'][-1,:,6], 'm', lw=2)
axes[1,3].set(xlabel='z (m)', ylabel='[I] (mol/m³)', title='Initiator profile')

plt.tight_layout()
plt.savefig('ldpe_pfr_v04_basecase.png', dpi=150)
print('\nFigure saved: ldpe_pfr_v04_basecase.png')

# %%
# ---- Figure 2: β-scission effect — Mn and PDI vs. T₀ ----------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle('v04 — β-scission effect  (Mn and Mw vs. inlet temperature)', fontsize=11)

T0_arr = np.array(T0_cases)
Mn_bs_arr  = np.array([T_sens_bs[t]['Mn_exit']   for t in T0_cases])
Mn_nbs_arr = np.array([T_sens_nobs[t]['Mn_exit'] for t in T0_cases])
Mw_bs_arr  = np.array([T_sens_bs[t]['Mw_exit']   for t in T0_cases])
Mw_nbs_arr = np.array([T_sens_nobs[t]['Mw_exit'] for t in T0_cases])
Tpk_arr    = np.array([T_sens_bs[t]['T_peak']    for t in T0_cases])
dMn_pct    = (Mn_nbs_arr - Mn_bs_arr) / Mn_nbs_arr * 100.

ax = axes2[0]
ax.plot(T0_arr, Mn_bs_arr,  'o-', lw=2, color='royalblue',   label='with β-scission')
ax.plot(T0_arr, Mn_nbs_arr, 's--',lw=2, color='tomato',      label='no β-scission (v03)')
ax.axhline(30,  color='gray', ls=':', lw=0.8)
ax.axhline(150, color='gray', ls=':', lw=0.8)
ax.set(xlabel='T₀ (°C)', ylabel='Mn (kg/mol)', title='Mn — β-scission reduces MW at high T')
ax.legend(fontsize=8)

ax = axes2[1]
ax.plot(T0_arr, Mw_bs_arr,  'o-', lw=2, color='royalblue',   label='with β-scission')
ax.plot(T0_arr, Mw_nbs_arr, 's--',lw=2, color='tomato',      label='no β-scission (v03)')
ax.axhline(100, color='gray', ls=':', lw=0.8)
ax.axhline(500, color='gray', ls=':', lw=0.8)
ax.set(xlabel='T₀ (°C)', ylabel='Mw (kg/mol)', title='Mw')
ax.legend(fontsize=8)

ax = axes2[2]
ax.bar(T0_arr, dMn_pct, width=1.2, color='steelblue', edgecolor='k', lw=0.8)
ax2b = ax.twinx()
ax2b.plot(T0_arr, Tpk_arr, 'r^-', ms=6, lw=1.5, label='T_peak')
ax2b.set_ylabel('T_peak (°C)', color='r')
ax2b.tick_params(axis='y', colors='r')
ax2b.axhline(250, color='r', ls=':', lw=0.8, alpha=0.5)
ax.set(xlabel='T₀ (°C)', ylabel='ΔMn β-scission effect (%)',
       title='Mn reduction from β-scission\n(dashed: T_peak = 250 °C threshold)')
ax.axhline(0, color='k', lw=0.5)

plt.tight_layout()
plt.savefig('ldpe_pfr_v04_bscission_effect.png', dpi=150)
print('Figure saved: ldpe_pfr_v04_bscission_effect.png')

# %%
# ---- Figure 3: Temperature sensitivity (β-scission ON) ---------------------
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))
fig3.suptitle('Temperature sensitivity  (T₀ = 180–210 °C, β-scission ON)', fontsize=11)
colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(T0_cases)))

for (T0c, r), col in zip(T_sens_bs.items(), colors):
    lbl = f'{T0c:.0f} °C'
    zr = r['z']
    axes3[0].plot(zr, r['T_z'],     color=col, label=lbl, lw=1.5)
    axes3[1].plot(zr, r['X_z']*100, color=col, label=lbl, lw=1.5)
    axes3[2].plot(zr, r['Mn_z'],    color=col, label=lbl, lw=1.5)
    axes3[3].plot(zr, r['PDI_z'],   color=col, label=lbl, lw=1.5)

for ax, yl, ttl, (lo, hi) in zip(axes3,
    ['T (°C)', 'X (%)', 'Mn (kg/mol)', 'PDI'],
    ['Reactor temperature', 'Conversion', 'Number-avg MW', 'PDI'],
    [(200,300), (5,15), (30,150), (3,15)]):
    ax.axhline(lo, color='gray', ls=':', lw=0.8)
    ax.axhline(hi, color='gray', ls=':', lw=0.8)
    ax.set(xlabel='z (m)', ylabel=yl, title=ttl)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('ldpe_pfr_v04_T_sensitivity.png', dpi=150)
print('Figure saved: ldpe_pfr_v04_T_sensitivity.png')

# %%
# ---- Figure 4: Spatial Mn profile with vs. without β-scission at T₀=210°C -
fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4))
fig4.suptitle('β-scission spatial effect  (T₀ = 210 °C, T_peak ≈ 270 °C)', fontsize=11)

r_hi_bs  = T_sens_bs[210.]
r_hi_nbs = T_sens_nobs[210.]
zr       = r_hi_bs['z']

axes4[0].plot(zr, r_hi_bs['T_z'],  'r',   lw=2, label='T (with β-sci.)')
axes4[0].plot(zr, r_hi_nbs['T_z'], 'r--', lw=1.5, label='T (no β-sci.)')
axes4[0].axhline(250, color='orange', ls=':', lw=1.2, label='250 °C threshold')
axes4[0].set(xlabel='z (m)', ylabel='T (°C)', title='Temperature profile')
axes4[0].legend(fontsize=8)

axes4[1].plot(zr, r_hi_bs['Mn_z'],  'b',   lw=2, label='with β-scission')
axes4[1].plot(zr, r_hi_nbs['Mn_z'], 'b--', lw=1.5, label='no β-scission (v03)')
axes4[1].axhline(30,  color='gray', ls=':', lw=0.8)
axes4[1].axhline(150, color='gray', ls=':', lw=0.8)
axes4[1].set(xlabel='z (m)', ylabel='Mn (kg/mol)', title='Mn spatial profile')
axes4[1].legend(fontsize=8)

axes4[2].plot(zr, r_hi_bs['Mw_z'],  'g',   lw=2, label='with β-scission')
axes4[2].plot(zr, r_hi_nbs['Mw_z'], 'g--', lw=1.5, label='no β-scission (v03)')
axes4[2].axhline(100, color='gray', ls=':', lw=0.8)
axes4[2].axhline(500, color='gray', ls=':', lw=0.8)
axes4[2].set(xlabel='z (m)', ylabel='Mw (kg/mol)', title='Mw spatial profile')
axes4[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('ldpe_pfr_v04_bscission_spatial.png', dpi=150)
print('Figure saved: ldpe_pfr_v04_bscission_spatial.png')

# %%
# ---- QSSA analytical check -------------------------------------------------
print('\n--- QSSA + β-scission check at T₀=195 °C ---')
T_K  = 195. + 273.15
kd_  = arrhenius(A_kd,   Ea_kd,   T_K)
kp_  = arrhenius(A_kp,   Ea_kp,   T_K)
ktc_ = arrhenius(A_ktc,  Ea_ktc,  T_K)
ktd_ = arrhenius(A_ktd,  Ea_ktd,  T_K)
ktrm_= arrhenius(A_ktrm, Ea_ktrm, T_K)
kbs_ = arrhenius(A_kbs,  Ea_kbs,  T_K)
l0_  = np.sqrt(2.*f_eff*kd_*ini_0 / (ktc_+ktd_))
DPn_ = kp_*mono_0 / (ktrm_*mono_0 + 0.5*(ktc_+ktd_)*l0_)
Mn_  = DPn_ * Mw_mono / 1000.
Cm_  = ktrm_/kp_
# β-scission characteristic time vs. residence time
tau_  = L / v
tau_bs = 1. / kbs_ if kbs_ > 0 else np.inf
print(f'  kd        = {kd_:.4e} s⁻¹   (t½ = {np.log(2)/kd_:.1f} s)')
print(f'  kp        = {kp_:.4f} m³/(mol·s)')
print(f'  kbs       = {kbs_:.4e} s⁻¹   (τ_bs = {tau_bs:.0f} s)')
print(f'  τ_res     = {tau_:.1f} s    τ_bs/τ_res = {tau_bs/tau_:.1f}  '
      f'→ β-scission {"negligible" if tau_bs/tau_ > 20 else "significant"} at 195 °C')
print(f'  Cm_eff    = ktrm/kp = {Cm_:.4e}')
print(f'  DPn_QSSA  = {DPn_:.0f}    Mn_QSSA = {Mn_:.2f} kg/mol')

print('\n--- β-scission significance at higher T ---')
for Tc_ in [230., 250., 270., 300.]:
    T_K2  = Tc_ + 273.15
    kbs2_ = arrhenius(A_kbs, Ea_kbs, T_K2)
    tb2   = 1./kbs2_ if kbs2_ > 0 else np.inf
    rel   = tau_/tb2
    print(f'  {Tc_:.0f} °C : k_bs={kbs2_:.3e} s⁻¹  τ_bs={tb2:.1f} s  '
          f'τ_res/τ_bs={rel:.3f}  → '
          f'{"significant" if rel > 0.05 else "minor" if rel > 0.01 else "negligible"}')

plt.show()

# %%
# =============================================================================
# Physical basis for v04 changes
# =============================================================================
#
# v04-1  β-scission mechanism in LDPE:
#        In industrial LDPE tubular reactors, mid-chain radicals (MCR) are
#        formed when a growing chain radical (Pₙ) abstracts a hydrogen atom
#        from a dead polymer chain (Dₘ) — intermolecular chain transfer to
#        polymer (ktrp). The resulting tertiary radical (MCR) on the backbone
#        of Dₘ can either:
#          (a) propagate (add monomer) — creates a long-chain branch (LCB)
#          (b) undergo β-scission — cleaves the backbone into two pieces
#        The β-scission product is one shorter dead chain (vinyl terminus) +
#        one new primary living radical.
#
#        This model does NOT track MCR as a separate species (would add 3 more
#        moment equations). Instead, β-scission is treated as a lumped first-
#        order process acting on the total living radical moments, with
#        effective Arrhenius parameters (A_kbs=1e11, Ea_kbs=130 kJ/mol)
#        chosen so that the effect is:
#          • negligible at 195 °C (τ_bs/τ_res >> 1)
#          • minor at 250 °C     (~4 % Mn reduction over the hot-spot zone)
#          • significant at 280+ °C (~10 % and growing)
#
#        Moment equations derived from population balance with symmetric
#        random-scission approximation (uniform fragment length m ∈ [1, n-1]):
#          dλ₁/dt|_bs = −k_bs·λ₁/2        (live radicals shorten)
#          dλ₂/dt|_bs = −k_bs·λ₂·2/3      (second moment decreases)
#          dμ₀/dt|_bs = +k_bs·λ₀           one new dead chain per event
#          dμ₁/dt|_bs = +k_bs·λ₁/2        dead fragment average length = n/2
#          dμ₂/dt|_bs = +k_bs·λ₂/3        dead fragment second moment
#
#        Note: λ₀ is unchanged (one live radical consumed, one created).
#        Note: μ₁ and μ₂ contributions from β-scission do NOT require the μ₃
#        closure because they are expressed in terms of λ moments (live chain
#        averages), not dead chain moments.
#
#        Literature references for k_β values in LDPE MOM models:
#          Pladis & Kiparissides, AIChE J. 44(9), 2024–2039 (1998)
#          Brandolin et al., Polym. React. Eng. 4(1-2), 1–47 (1996)
#          Kim & Iedema, Chem. Eng. Sci. 59(10), 2039–2052 (2004)
# =============================================================================
