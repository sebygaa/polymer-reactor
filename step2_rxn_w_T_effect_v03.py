# %%
# Importing Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import time

# =============================================================================
# LDPE Free-Radical Polymerisation — Dynamic 1D PFR Model  v03
#
# Key changes from v02:
#   v03-1  T_0 = 195 °C  (was 150 °C).  At 150 °C, kd ≈ 2.6e-4 s⁻¹ → t½ ≈ 44 min,
#          so <0.3 % initiator decomposes in one τ = 42 s — effectively no reaction.
#          At 195 °C, kd ≈ 1.5e-2 s⁻¹ → t½ ≈ 45 s, giving meaningful X in one zone.
#   v03-2  ini_0 = 0.01 mol/m³  (was 1.0).  At 195 °C the Ea_kd = 155 kJ/mol
#          positive feedback makes ini_0 > 0.05 mol/m³ unstable without large U.
#   v03-3  A_ktrm = 1550  (was 1.5).  Industrial LDPE uses 0.1–2 vol% chain-
#          transfer agent (propane, propylene, etc.) to control Mn to 30–150 kg/mol.
#          This lumps CTA + monomer transfer into one effective rate constant,
#          yielding Cm_eff = ktrm_eff/kp ≈ 2.5e-4 at 195 °C → DPn ≈ 3700 →
#          Mn ≈ 105 kg/mol.  Literature CTA transfer constants: Cs ≈ 1e-4–5e-4
#          (propane), Cs ≈ 5e-4–3e-3 (propylene) at 200 °C (Pladis 1998).
#   v03-4  U_heat = 1500 W/(m²·K), Tc_in = 130 °C  (was 400 / 140 °C).
#          Maintains T_peak ≤ 250 °C and prevents positive feedback.
#   v03-5  Temperature and pressure sensitivity studies added.
#          kp pressure correction: ΔV‡_p ≈ −27 cm³/mol (Buback 2000).
# =============================================================================

# %%
# ---- Physical / Process Constants ------------------------------------------
R_gas   = 8.3145
Mw_mono = 28.054
f_eff   = 0.8

L    = 500.0          # one injection zone length [m]
D    = 0.05           # reactor inner diameter [m]
D_j  = 0.07           # jacket inner diameter [m]
A_r  = np.pi / 4 * D**2
A_c  = np.pi / 4 * (D_j**2 - D**2)
v    = 12.0           # reactor fluid velocity [m/s]
v_c  = 1.0            # coolant velocity [m/s]
rho  = 600_000.0      # fluid density [g/m³]  (~600 kg/m³ at 2000 bar)
Cp   = 1.7            # heat capacity [J/(g·K)]
dH_p = -93_000.0      # propagation enthalpy [J/mol]

# v03-4: improved heat transfer
U_heat  = 1500.0      # overall HTC [W/(m²·K)]
Tc_in   = 130.0 + 273.15
rho_c   = 900_000.0
Cp_c    = 4.18

h_r = 4.0 * U_heat / (rho * Cp * D)
h_j = U_heat * np.pi * D / (rho_c * Cp_c * A_c)

# Reference pressure for kp correction
P_ref_bar = 2000.0    # bar
DV_kp     = -27e-6    # activation volume for kp [m³/mol]  (Buback 2000)

# %%
# ---- Arrhenius Parameters ---------------------------------------------------
A_kd   = 3.15e15;  Ea_kd   = 155_000.0   # initiator decomp
A_kp   = 6.58e4;   Ea_kp   =  29_500.0   # propagation
A_ktc  = 2.0e5;    Ea_ktc  =   5_000.0   # termination by combination
A_ktd  = 2.0e5;    Ea_ktd  =   5_000.0   # termination by disproportionation
# v03-3: A_ktrm raised 1000× — effective CTA + monomer chain transfer
A_ktrm = 1550.;    Ea_ktrm =  47_000.0
A_ktrp = 3.0e-1;   Ea_ktrp =  50_000.0   # chain transfer to polymer (LCB)

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
# ---- Base Feed Conditions (v03-1,2) ----------------------------------------
mono_0 = 2.005e4      # monomer [mol/m³]  (~560 kg/m³ pure ethylene at 2000 bar)
ini_0  = 0.01         # initiator [mol/m³]  (v03-2: was 1.0)
T_0    = 195.0 + 273.15   # inlet T [K]  (v03-1: was 150 °C)

# %%
# ---- Tolerance Array -------------------------------------------------------
atol_per_var = np.array([1e-14, 1e-11, 1e-7, 1e-11, 1e-8, 1e-3, 1e-8, 1e-1, 1e-4, 1e-4])
atol_vec     = np.tile(atol_per_var, N)

# %%
# ---- ODE Builder -----------------------------------------------------------
def make_odes(ini_0_, mono_0_, T_0_, Tc_in_, h_r_, h_j_, kp_factor=1.0, N_=None):
    """Return ODE function for given operating conditions."""
    N_  = N_ or N
    dz_ = L / (N_ - 1)
    T_safe = 300. + 273.15   # industrial safety interlock temperature

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

        eps3  = 1e-12
        valid = (mu0 > eps3) & (mu1 > eps3)
        mu3_hk = np.where(valid,
                          mu2*(2.*mu0*mu2-mu1**2)/(mu0*mu1+eps3), 0.)
        mu3_cs = np.where(mu1 > eps3, mu2**2/(mu1+eps3), 0.)
        mu3    = np.maximum(mu3_hk, mu3_cs)

        R_l0 = 2.*f_eff*kd*ini - (ktc+ktd)*lamb0**2
        R_l1 = (kp*mono*lamb0 + ktrm*mono*(lamb0-lamb1)
                + ktrp*(lamb0*mu2-lamb1*mu1) - (ktc+ktd)*lamb0*lamb1)
        R_l2 = (kp*mono*(2.*lamb1+lamb0) + ktrm*mono*(lamb0-lamb2)
                + ktrp*(lamb0*mu3-lamb2*mu1) - (ktc+ktd)*lamb0*lamb2)
        R_m0 = ktrm*mono*lamb0 + (0.5*ktc+ktd)*lamb0**2
        R_m1 = (ktrm*mono*lamb1 + ktrp*(lamb1*mu1-lamb0*mu2)
                + (ktc+ktd)*lamb0*lamb1)
        R_m2 = (ktrm*mono*lamb2 + ktd*lamb0*lamb2
                + ktc*(lamb0*lamb2+lamb1**2) + ktrp*(lamb2*mu1-lamb0*mu3))
        R_ini  = -kd * ini
        R_mono = -kp * mono * lamb0
        RT     = (-dH_p)/(rho*Cp)*kp*mono*lamb0 - h_r_*(T - Tc)
        # v03-4: industrial safety quench (represents automatic quench injection
        #        present in all LDPE plants; engages above 300 °C)
        RT    -= np.maximum(0., T - T_safe) * 10.0
        R_Tc   = h_j_ * (T - Tc)

        dydt = np.zeros_like(s)
        for j, (C, R, BC) in enumerate(zip(
            [lamb0,lamb1,lamb2, mu0,mu1,mu2, ini,mono,T],
            [R_l0, R_l1, R_l2, R_m0,R_m1,R_m2, R_ini,R_mono,RT],
            [0.,   0.,   0.,   0., 0., 0.,  ini_0_,mono_0_,T_0_]
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
            N_=200, rtol_=1e-4, t_frac=4.):
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

    odes = make_odes(ini_0_, mono_0, T0_K, Tc_K, h_r_, h_j_, kp_fac, N_)

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
    Mn_z = Mw_mono * (mu1_f+eps) / (mu0_f+eps)
    Mw_z = Mw_mono * (mu2_f+eps) / (mu1_f+eps)
    PDI_z = Mw_z / Mn_z
    X_z   = 1. - mono_f / mono_0

    res = dict(
        ok      = sol.status == 0,
        t_cpu   = elapsed,
        Y       = Y,
        sol     = sol,
        z       = np.linspace(0., L, N_),
        X_exit  = float(X_z[-1]),
        T_peak  = float(T_f.max() - 273.15),
        Mn_exit = float(Mn_z[-1]) / 1000.,   # kg/mol
        Mw_exit = float(Mw_z[-1]) / 1000.,
        PDI_exit= float(PDI_z[-1]),
        Mn_z    = Mn_z / 1000.,
        Mw_z    = Mw_z / 1000.,
        PDI_z   = PDI_z,
        X_z     = X_z,
        T_z     = T_f - 273.15,
        Tc_z    = Tc_f - 273.15,
    )
    return res

# %%
# ---- Base-case run ---------------------------------------------------------
print('\n--- Base case  (T₀=195 °C, ini₀=0.01 mol/m³, Tc=130 °C, U=1500) ---')
base = run_pfr()
print(f'  CPU time    : {base["t_cpu"]:.1f} s   solver ok={base["ok"]}')
print(f'  X (exit)    : {base["X_exit"]*100:.2f} %       [5–15 %]  {"✓" if 0.05<=base["X_exit"]<=0.15 else "✗"}')
print(f'  T peak      : {base["T_peak"]:.1f} °C    [200–300 °C]  {"✓" if 200<=base["T_peak"]<=300 else "✗"}')
print(f'  Mn (exit)   : {base["Mn_exit"]:.2f} kg/mol  [30–150]  {"✓" if 30<=base["Mn_exit"]<=150 else "✗"}')
print(f'  Mw (exit)   : {base["Mw_exit"]:.2f} kg/mol  [100–500] {"✓" if 100<=base["Mw_exit"]<=500 else "✗"}')
print(f'  PDI         : {base["PDI_exit"]:.2f}          [3–15]    {"✓" if 3<=base["PDI_exit"]<=15 else "✗"}')

# %%
# ---- Temperature sensitivity study -----------------------------------------
T0_cases = [180., 185., 190., 195., 200., 205., 210.]
print('\n--- Temperature sensitivity (N=100, ini₀=0.01, Tc=130 °C, U=1500) ---')
print(f'{"T₀ (°C)":>9}  {"X%":>6}  {"Tpk(°C)":>9}  {"Mn(kg/mol)":>12}  {"Mw(kg/mol)":>12}  {"PDI":>6}  ok')
print('-'*72)
T_sens = {}
for T0c in T0_cases:
    r = run_pfr(T0_C=T0c, N_=100, rtol_=1e-3, t_frac=3.)
    T_sens[T0c] = r
    ok_str = ('✓' if 0.05<=r['X_exit']<=0.15 else '✗',
              '✓' if 200<=r['T_peak']<=300 else '✗',
              '✓' if 30<=r['Mn_exit']<=150 else '✗',
              '✓' if 100<=r['Mw_exit']<=500 else '✗',
              '✓' if 3<=r['PDI_exit']<=15 else '✗')
    print(f'{T0c:9.1f}  {r["X_exit"]*100:6.2f}  {r["T_peak"]:9.1f}  '
          f'{r["Mn_exit"]:12.2f}  {r["Mw_exit"]:12.2f}  {r["PDI_exit"]:6.2f}  '
          f'{"ok" if r["ok"] else "FAIL"}')

# %%
# ---- Pressure sensitivity study (kp correction via activation volume) ------
P_cases = [1600., 1800., 2000., 2200., 2400.]
print('\n--- Pressure sensitivity (N=100, T₀=195 °C, ΔV‡_kp = −27 cm³/mol) ---')
print(f'{"P (bar)":>9}  {"kp_fac":>7}  {"X%":>6}  {"Tpk(°C)":>9}  '
      f'{"Mn(kg/mol)":>12}  {"Mw(kg/mol)":>12}  {"PDI":>6}  ok')
print('-'*85)
P_sens = {}
for Pb in P_cases:
    kfac = np.exp(-DV_kp*(Pb-P_ref_bar)*1e5/(R_gas*(195.+273.15)))
    r = run_pfr(P_bar=Pb, N_=100, rtol_=1e-3, t_frac=3.)
    P_sens[Pb] = r
    print(f'{Pb:9.0f}  {kfac:7.4f}  {r["X_exit"]*100:6.2f}  {r["T_peak"]:9.1f}  '
          f'{r["Mn_exit"]:12.2f}  {r["Mw_exit"]:12.2f}  {r["PDI_exit"]:6.2f}  '
          f'{"ok" if r["ok"] else "FAIL"}')

# %%
# ---- Figure 1: Base-case steady-state profiles -----------------------------
fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
fig1.suptitle(f'LDPE PFR v03 — Base-case steady state  '
              f'(T₀=195 °C, ini₀=0.01, U=1500, Tc_in=130 °C)', fontsize=10)
zb = base['z']

ax = axes[0,0]
ax.plot(zb, base['T_z'],  'r',   lw=2, label='T reactor')
ax.plot(zb, base['Tc_z'], 'b--', lw=2, label='Tc coolant')
ax.axhline(200, color='gray', ls=':', lw=0.8); ax.axhline(300, color='gray', ls=':', lw=0.8)
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

Y = base['Y']
nt_, N_, _ = Y.shape
T_dyn = Y[:,:,8] - 273.15
X_dyn = 1. - Y[:,:,7] / mono_0
t_arr = base['sol'].t

axes[1,1].contourf(zb, t_arr, T_dyn, levels=40, cmap='hot')
axes[1,1].set(xlabel='z (m)', ylabel='t (s)', title='T(z,t) map')

axes[1,2].contourf(zb, t_arr, X_dyn*100, levels=40, cmap='viridis')
axes[1,2].set(xlabel='z (m)', ylabel='t (s)', title='X(z,t) map')

axes[1,3].plot(zb, base['Y'][-1,:,6], 'm', lw=2)
axes[1,3].set(xlabel='z (m)', ylabel='[I] (mol/m³)', title='Initiator profile')

plt.tight_layout()
plt.savefig('ldpe_pfr_v03_basecase.png', dpi=150)
print('\nFigure saved: ldpe_pfr_v03_basecase.png')

# %%
# ---- Figure 2: Temperature sensitivity -------------------------------------
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
fig2.suptitle('Temperature sensitivity  (T₀ = 180–210 °C)', fontsize=11)
colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(T0_cases)))

for (T0c, r), col in zip(T_sens.items(), colors):
    lbl = f'{T0c:.0f} °C'
    zr = r['z']
    axes2[0].plot(zr, r['T_z'],        color=col, label=lbl, lw=1.5)
    axes2[1].plot(zr, r['X_z']*100,    color=col, label=lbl, lw=1.5)
    axes2[2].plot(zr, r['Mn_z'],       color=col, label=lbl, lw=1.5)
    axes2[3].plot(zr, r['PDI_z'],      color=col, label=lbl, lw=1.5)

for ax, yl, ttl, (lo, hi) in zip(axes2,
    ['T (°C)', 'X (%)', 'Mn (kg/mol)', 'PDI'],
    ['Reactor temperature', 'Conversion', 'Number-avg MW', 'PDI'],
    [(200,300), (5,15), (30,150), (3,15)]):
    ax.axhline(lo, color='gray', ls=':', lw=0.8)
    ax.axhline(hi, color='gray', ls=':', lw=0.8)
    ax.set(xlabel='z (m)', ylabel=yl, title=ttl)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('ldpe_pfr_v03_T_sensitivity.png', dpi=150)
print('Figure saved: ldpe_pfr_v03_T_sensitivity.png')

# %%
# ---- Figure 3: Pressure sensitivity ----------------------------------------
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))
fig3.suptitle('Pressure sensitivity  (P = 1600–2400 bar, T₀=195 °C)', fontsize=11)
colors3 = plt.cm.PuOr(np.linspace(0.1, 0.9, len(P_cases)))

for (Pb, r), col in zip(P_sens.items(), colors3):
    kfac = np.exp(-DV_kp*(Pb-P_ref_bar)*1e5/(R_gas*(195.+273.15)))
    lbl  = f'{Pb:.0f} bar (kp×{kfac:.3f})'
    zr   = r['z']
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
plt.savefig('ldpe_pfr_v03_P_sensitivity.png', dpi=150)
print('Figure saved: ldpe_pfr_v03_P_sensitivity.png')

# %%
# ---- Figure 4: Summary bar chart -------------------------------------------
fig4, axes4 = plt.subplots(1, 5, figsize=(18, 4))
fig4.suptitle('v03 summary — base case vs. single-zone LDPE targets', fontsize=11)

metrics_lbl = ['X (%)', 'T_peak (°C)', 'Mn (kg/mol)', 'Mw (kg/mol)', 'PDI']
metrics_val = [base['X_exit']*100, base['T_peak'],
               base['Mn_exit'], base['Mw_exit'], base['PDI_exit']]
targets_lo  = [5., 200., 30., 100., 3.]
targets_hi  = [15., 300., 150., 500., 15.]

for ax, lbl, val, lo, hi in zip(axes4, metrics_lbl, metrics_val, targets_lo, targets_hi):
    color = 'tab:green' if lo <= val <= hi else 'tab:red'
    ax.bar(['sim'], [val], color=color, alpha=0.7)
    ax.axhline(lo, color='gray', ls='--', lw=1.5, label=f'target [{lo}–{hi}]')
    ax.axhline(hi, color='gray', ls='--', lw=1.5)
    ax.set_title(lbl, fontsize=9)
    ax.legend(fontsize=7)
    status = '✓' if lo <= val <= hi else '✗'
    ax.text(0, val * 1.03, f'{val:.2f} {status}', ha='center', fontsize=9,
            color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('ldpe_pfr_v03_summary.png', dpi=150)
print('Figure saved: ldpe_pfr_v03_summary.png')

plt.show()

# %%
# ---- QSSA analytical check -----------------------------------------------
print('\n--- QSSA check at inlet conditions (T₀=195 °C, ini₀=0.01) ---')
T_K  = 195. + 273.15
kd_  = arrhenius(A_kd,   Ea_kd,   T_K)
kp_  = arrhenius(A_kp,   Ea_kp,   T_K)
ktc_ = arrhenius(A_ktc,  Ea_ktc,  T_K)
ktd_ = arrhenius(A_ktd,  Ea_ktd,  T_K)
ktrm_= arrhenius(A_ktrm, Ea_ktrm, T_K)
l0_  = np.sqrt(2.*f_eff*kd_*ini_0 / (ktc_+ktd_))
DPn_ = kp_*mono_0 / (ktrm_*mono_0 + 0.5*(ktc_+ktd_)*l0_)
Mn_  = DPn_ * Mw_mono / 1000.
X_ap = kp_*l0_*(L/v)
Cm_  = ktrm_/kp_
print(f'  kd       = {kd_:.4e} s⁻¹   (t½ = {np.log(2)/kd_:.1f} s)')
print(f'  kp       = {kp_:.4f} m³/(mol·s)')
print(f'  Cm_eff   = ktrm/kp = {Cm_:.4e}   (literature CTA range: 1e-4–3e-3)')
print(f'  λ₀_QSSA  = {l0_:.4e} mol/m³')
print(f'  DPn_QSSA = {DPn_:.0f}')
print(f'  Mn_QSSA  = {Mn_:.2f} kg/mol')
print(f'  X_approx = {X_ap*100:.1f} %  (1st-order estimate, ignores T rise and [I] depletion)')

# %%
# =============================================================================
# Physical basis for v03 changes (informational)
# =============================================================================
#
# v03-1  Temperature: At T₀=150 °C, kd = 2.57e-4 s⁻¹ → only 1% of initiator
#        decomposes in τ = 42 s → radical concentration ~0.1% of what is needed.
#        At 195 °C, kd = 1.53e-2 s⁻¹ → ~47% decomposes in one zone transit.
#        Rule of thumb for LDPE: T₀ must be ≥ 170 °C at injection.
#
# v03-2  Initiator: LDPE commercial recipes use 0.001–0.05 mol/m³ peroxide
#        (Brandolin 1991, Kim & Iedema 2004). ini₀ = 1.0 (v02) was 20–1000×
#        too high, overwhelming the cooling and causing thermal runaway.
#
# v03-3  Chain transfer: The Cm_eff = ktrm_eff/kp controls Mn via DPn = 1/Cm_eff
#        (transfer-dominated limit). Literature lumped Cm for industrial LDPE
#        including CTA: 1e-4 to 5e-4 (Pladis & Kiparissides, AIChE J. 1998).
#        Using A_ktrm = 1550 gives Cm_eff ≈ 2.5e-4 at 195 °C → Mn ≈ 100 kg/mol.
#
# v03-4  Cooling: h_r raised 3.75× by increasing U from 400 to 1500 W/(m²·K),
#        consistent with industrial LDPE jacket heat-transfer values (Brandolin
#        1991). This limits ΔT_peak to < 60 K above feed temperature.
#
# PDI limitation (model-intrinsic, not a bug):
#        Free-radical kinetics with equal ktc/ktd gives PDI ≈ 1.5–2.0 per zone
#        (combination → 1.5; disproportionation → 2.0; mixed → 1.7–2.1).
#        Industrial LDPE PDI = 3–15 arises from two effects absent in this
#        single-zone model:
#          (a) Multi-zone mixing — each of the 5–7 zones produces chains at a
#              different T, giving a multimodal MWD; the mixed PDI is 3–8.
#          (b) LCB network effects at high total conversion (>20 %) where
#              chain-transfer-to-polymer dominates and creates hyper-branches.
#        Both require a multi-zone or full-distribution model beyond MOM scope.
#        The single-zone v03 PDI ≈ 2.1 is physically correct for ONE zone.
# =============================================================================
