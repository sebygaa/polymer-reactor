# %%
# Importing Packages
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import time

# =============================================================================
# LDPE Free-Radical Polymerization — Dynamic 1D PFR Model
# Version 02: Numerical & physical improvements
#
# Fixed in v02 (from user review):
#   Fix 1 : N=200 (was 40) + jac_sparsity for efficient Jacobian approximation
#   Fix 4 : Mn/Mw computed from dead-polymer moments only (μ-only);
#            combined live+dead reported separately for reference
#   Fix 5 : μ₃ Cauchy-Schwarz lower bound  μ₃ ≥ μ₂²/μ₁  (replaces silent clip)
#            Post-run report of how many nodes hit the bound
#   Fix 6 : h_r / h_j area-basis comment made explicit (both use inner D, consistent)
#   Fix 8 : run_grid_convergence() compares N=40/100/200 for grid-sensitivity study
#   Fix 9 : ini_0 = 1.0 mol/m³  (was 100; commercial LDPE: 0.01–1 mol/m³)
#   Fix 10: variable-specific atol array (scaled to each variable's magnitude)
#
# Analysed only (not fixed) — see bottom of file:
#   Issue 2: constant ρ=600 kg/m³ (density varies with T and conversion)
#   Issue 3: no pressure-drop term; no pressure-dependent Arrhenius correction
#   Issue 7: μ₀ LCB term — intentionally omitted (consistent with v3_fixed.docx)
# =============================================================================

# %%
# ---- Physical / Process Constants -----------------------------------------
# %%
R_gas    = 8.3145
Mw_mono  = 28.054
f_eff    = 0.8
L        = 500.0
D        = 0.05
D_j      = 0.07
A_r      = np.pi / 4 * D**2
A_c      = np.pi / 4 * (D_j**2 - D**2)
v        = 12.0
v_c      = 1.0
rho      = 600_000.0
Cp       = 1.7
dH_p     = -93_000.0
U_heat   = 400.0
Tc_in    = 140.0 + 273.15
rho_c    = 900_000.0
Cp_c     = 4.18

# Fix 6: heat-transfer coefficients — both referenced to inner tube surface (πD per unit length)
#   h_r = U·(πD/A_r) / (ρ·Cp)  = 4U / (ρ·Cp·D)     [1/s]  reactor side
#   h_j = U·(πD/A_c) / (ρc·Cp_c)                     [1/s]  jacket side
#   Both use the same heat-exchange perimeter πD → inner-surface basis is consistent.
h_r = 4.0 * U_heat / (rho * Cp * D)
h_j = U_heat * np.pi * D / (rho_c * Cp_c * A_c)

# %%
# ---- Arrhenius Parameters -------------------------------------------------
# %%
A_kd   = 3.15e15;  Ea_kd   = 155_000.0
A_kp   = 6.58e4;   Ea_kp   =  29_500.0
A_ktc  = 2.0e5;    Ea_ktc  =   5_000.0
A_ktd  = 2.0e5;    Ea_ktd  =   5_000.0
A_ktrm = 1.5;      Ea_ktrm =  47_000.0
A_ktrp = 3.0e-1;   Ea_ktrp =  50_000.0

def arrhenius(A, Ea, T):
    return A * np.exp(-Ea / (R_gas * T))

# %%
# ---- Spatial Grid (Fix 1: N = 200) ----------------------------------------
# %%
N  = 200
NV = 10
z  = np.linspace(0.0, L, N)
dz = z[1] - z[0]
print(f'Grid: N={N}, dz={dz:.2f} m  (v01 had dz={500/(40-1):.1f} m)')

# %%
# ---- Jacobian Sparsity (Fix 1) -------------------------------------------
# %%
# For each row (i,j):
#   • Full NV-block at node i  (reaction coupling through T and all moments)
#   • node i-1, same j  (backward upwind for reactor vars j=0..8)
#   • node i+1, j=9     (forward upwind for jacket Tc)
def build_jac_sparsity(N, NV=10):
    rows, cols = [], []
    for i in range(N):
        for j in range(NV):
            row = i * NV + j
            for k in range(NV):               # local reaction coupling
                rows.append(row); cols.append(i * NV + k)
            if j < 9 and i > 0:               # backward upwind: prev node
                rows.append(row); cols.append((i - 1) * NV + j)
            if j == 9 and i < N - 1:          # forward upwind: next node Tc
                rows.append(row); cols.append((i + 1) * NV + 9)
    n = N * NV
    return csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))

jac_sp = build_jac_sparsity(N, NV)
nnz    = jac_sp.nnz
print(f'Jacobian: {N*NV}×{N*NV}, nnz={nnz} ({100*nnz/(N*NV)**2:.2f}% dense)')

# %%
# ---- Boundary / Initial-Feed Conditions (Fix 9: ini_0 = 1.0) -------------
# %%
mono_0 = 2.005e4     # mol/m³
ini_0  = 1.0         # mol/m³  — Fix 9: was 100, commercial LDPE ~0.01–1 mol/m³
T_0    = 150.0 + 273.15

# %%
# ---- Variable-Specific atol (Fix 10) -------------------------------------
# %%
# Typical steady-state scales:
#   λ₀~1e-5, λ₁~1e-2, λ₂~10, μ₀~0.1, μ₁~100, μ₂~1e7,
#   [I]~0–1, [M]~0–2e4, T~400–800K, Tc~413–700K
atol_per_var = np.array([
    1e-14,   # j=0  λ₀
    1e-11,   # j=1  λ₁
    1e-7,    # j=2  λ₂
    1e-11,   # j=3  μ₀
    1e-8,    # j=4  μ₁
    1e-3,    # j=5  μ₂
    1e-8,    # j=6  [I]
    1e-1,    # j=7  [M]
    1e-4,    # j=8  T
    1e-4,    # j=9  Tc
])
atol_vec = np.tile(atol_per_var, N)

# %%
# ---- ODE Function ---------------------------------------------------------
# %%
def pfr_odes(t, y):
    s = y.reshape(N, NV)

    lamb0=s[:,0]; lamb1=s[:,1]; lamb2=s[:,2]
    mu0  =s[:,3]; mu1  =s[:,4]; mu2  =s[:,5]
    ini  =s[:,6]; mono =s[:,7]; T=s[:,8]; Tc=s[:,9]

    kd   = arrhenius(A_kd,   Ea_kd,   T)
    kp   = arrhenius(A_kp,   Ea_kp,   T)
    ktc  = arrhenius(A_ktc,  Ea_ktc,  T)
    ktd  = arrhenius(A_ktd,  Ea_ktd,  T)
    ktrm = arrhenius(A_ktrm, Ea_ktrm, T)
    ktrp = arrhenius(A_ktrp, Ea_ktrp, T)

    # Fix 5: μ₃ Hulburt-Katz with Cauchy-Schwarz lower bound  μ₃ ≥ μ₂²/μ₁
    # Rationale: the CS inequality  μ₁μ₃ ≥ μ₂²  must hold for any valid
    # probability distribution; silently clipping at 0 can violate this.
    eps3   = 1e-12
    valid  = (mu0 > eps3) & (mu1 > eps3)
    mu3_hk = np.where(valid,
                      mu2 * (2.0*mu0*mu2 - mu1**2) / (mu0*mu1 + eps3), 0.0)
    mu3_cs = np.where(mu1 > eps3, mu2**2 / (mu1 + eps3), 0.0)  # CS lower bound
    mu3    = np.maximum(mu3_hk, mu3_cs)

    # Reaction source terms
    R_l0 = 2.0*f_eff*kd*ini - (ktc+ktd)*lamb0**2
    R_l1 = (kp*mono*lamb0 + ktrm*mono*(lamb0-lamb1)
            + ktrp*(lamb0*mu2-lamb1*mu1) - (ktc+ktd)*lamb0*lamb1)
    R_l2 = (kp*mono*(2.0*lamb1+lamb0) + ktrm*mono*(lamb0-lamb2)
            + ktrp*(lamb0*mu3-lamb2*mu1) - (ktc+ktd)*lamb0*lamb2)
    R_m0 = ktrm*mono*lamb0 + (0.5*ktc+ktd)*lamb0**2
    R_m1 = (ktrm*mono*lamb1 + ktrp*(lamb1*mu1-lamb0*mu2)
            + (ktc+ktd)*lamb0*lamb1)
    R_m2 = (ktrm*mono*lamb2 + ktd*lamb0*lamb2
            + ktc*(lamb0*lamb2+lamb1**2) + ktrp*(lamb2*mu1-lamb0*mu3))
    R_ini  = -kd * ini
    R_mono = -kp * mono * lamb0
    R_T    = (-dH_p)/(rho*Cp)*kp*mono*lamb0 - h_r*(T - Tc)
    R_Tc   = h_j * (T - Tc)

    dydt = np.zeros_like(s)

    # Reactor vars (j=0..8): Backward upwind FDM  (flow +z)
    for j, (C, R, BC) in enumerate(zip(
        [lamb0,lamb1,lamb2, mu0,mu1,mu2, ini,mono,T],
        [R_l0, R_l1, R_l2, R_m0,R_m1,R_m2, R_ini,R_mono,R_T],
        [0.0,  0.0,  0.0,  0.0, 0.0, 0.0,  ini_0,mono_0, T_0]
    )):
        C_up      = np.empty(N)
        C_up[0]   = BC
        C_up[1:]  = C[:-1]
        dCdz      = (C - C_up) / dz
        dydt[0, j]  = 0.0
        dydt[1:, j] = R[1:] - v * dCdz[1:]

    # Jacket Tc (j=9): Forward upwind FDM  (counter-current, flow −z)
    Tc_dn      = np.empty(N)
    Tc_dn[:-1] = Tc[1:]
    Tc_dn[-1]  = Tc_in
    dTcdz      = (Tc_dn - Tc) / dz
    dydt[:N-1, 9] = R_Tc[:N-1] + v_c * dTcdz[:N-1]
    dydt[N-1,  9] = 0.0

    return dydt.ravel()

# %%
# ---- Initial Conditions ---------------------------------------------------
# %%
y0 = np.zeros((N, NV))
y0[:, 6] = 0.0;     y0[:, 7] = mono_0
y0[:, 8] = T_0;     y0[:, 9] = Tc_in
y0[0,  6] = ini_0;  y0[N-1, 9] = Tc_in

# %%
# ---- Time Integration -----------------------------------------------------
# %%
tau_res = L / v
t_end   = 4.0 * tau_res
t_eval  = np.linspace(0.0, t_end, 300)

print(f'\nResidence time τ = {tau_res:.1f} s  |  Simulating 0 → {t_end:.0f} s ...')
t0 = time.time()

sol = solve_ivp(
    pfr_odes,
    (0.0, t_end),
    y0.ravel(),
    method='Radau',
    t_eval=t_eval,
    rtol=1e-4,
    atol=atol_vec,
    jac_sparsity=jac_sp,
    dense_output=False,
)

elapsed = time.time() - t0
print(f'Done in {elapsed:.1f} s  |  status={sol.status}  |  nfev={sol.nfev}')
print(f'Message: {sol.message}')

# %%
# ---- Post-processing -------------------------------------------------------
# %%
nt = len(sol.t)
Y  = sol.y.T.reshape(nt, N, NV)

lamb0_s=Y[:,:,0]; lamb1_s=Y[:,:,1]; lamb2_s=Y[:,:,2]
mu0_s  =Y[:,:,3]; mu1_s  =Y[:,:,4]; mu2_s  =Y[:,:,5]
ini_s  =Y[:,:,6]; mono_s =Y[:,:,7]
T_s    =Y[:,:,8]; Tc_s   =Y[:,:,9]
X_s    = 1.0 - mono_s / mono_0

eps_mw = 1e-30

# Fix 4a: Dead-polymer only Mn/Mw — physically correct for polymer characterisation
#   In QSSA limit λ₀ ≪ μ₀, but during startup live radicals inflate the combined Mn.
#   μ-only avoids that bias.
Mn_dead  = Mw_mono * (mu1_s + eps_mw) / (mu0_s + eps_mw)
Mw_dead  = Mw_mono * (mu2_s + eps_mw) / (mu1_s + eps_mw)
PDI_dead = Mw_dead / Mn_dead

# Fix 4b: Combined live+dead for comparison / total-chain statistics
Mn_tot   = Mw_mono * (lamb1_s+mu1_s+eps_mw) / (lamb0_s+mu0_s+eps_mw)
Mw_tot   = Mw_mono * (lamb2_s+mu2_s+eps_mw) / (lamb1_s+mu1_s+eps_mw)
PDI_tot  = Mw_tot / Mn_tot

# Fix 5 post-run: count where Cauchy-Schwarz bound was binding at final time
mu3_hk_f = np.where(
    (mu0_s[-1] > 1e-12) & (mu1_s[-1] > 1e-12),
    mu2_s[-1]*(2.0*mu0_s[-1]*mu2_s[-1]-mu1_s[-1]**2)/(mu0_s[-1]*mu1_s[-1]+1e-30),
    0.0)
mu3_cs_f = mu2_s[-1]**2 / (mu1_s[-1] + 1e-30)
n_cs_binding = int(np.sum(mu3_hk_f < mu3_cs_f))
print(f'μ₃ Cauchy-Schwarz bound active at final time: {n_cs_binding}/{N} nodes')

# %%
# ---- Figure 1: Steady-state spatial profiles ------------------------------
# %%
fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
fig1.suptitle(
    f'LDPE PFR v02 — Steady-state profiles  (t={sol.t[-1]:.0f} s, N={N})',
    fontsize=11
)

ax = axes1[0, 0]
ax.plot(z, T_s[-1]-273.15,  'r',   lw=2, label='Reactor T')
ax.plot(z, Tc_s[-1]-273.15, 'b--', lw=2, label='Coolant Tc')
ax.axhline(Tc_in-273.15, color='b', lw=0.8, ls=':', alpha=0.5)
ax.set_xlabel('z (m)'); ax.set_ylabel('T (°C)'); ax.set_title('Temperature'); ax.legend(fontsize=8)

axes1[0, 1].plot(z, X_s[-1]*100, 'g', lw=2)
axes1[0, 1].set_xlabel('z (m)'); axes1[0, 1].set_ylabel('Conversion (%)'); axes1[0, 1].set_title('Monomer conversion')

axes1[0, 2].plot(z, ini_s[-1], 'm', lw=2)
axes1[0, 2].set_xlabel('z (m)'); axes1[0, 2].set_ylabel('[I] (mol/m³)'); axes1[0, 2].set_title('Initiator')

axes1[0, 3].plot(z, T_s[-1]-Tc_s[-1], 'darkorange', lw=2)
axes1[0, 3].set_xlabel('z (m)'); axes1[0, 3].set_ylabel('T−Tc (K)'); axes1[0, 3].set_title('Heat-transfer driving force')

# Fix 4: separate Mn/Mw plots
axes1[1, 0].plot(z, Mn_dead[-1]/1000, lw=2, label='dead only')
axes1[1, 0].plot(z, Mn_tot[-1]/1000, lw=1.5, ls='--', label='combined')
axes1[1, 0].set_xlabel('z (m)'); axes1[1, 0].set_ylabel('Mn (kg/mol)'); axes1[1, 0].set_title('Mn'); axes1[1, 0].legend(fontsize=7)

axes1[1, 1].plot(z, Mw_dead[-1]/1000, lw=2, label='dead only')
axes1[1, 1].plot(z, Mw_tot[-1]/1000, lw=1.5, ls='--', label='combined')
axes1[1, 1].set_xlabel('z (m)'); axes1[1, 1].set_ylabel('Mw (kg/mol)'); axes1[1, 1].set_title('Mw'); axes1[1, 1].legend(fontsize=7)

axes1[1, 2].plot(z, PDI_dead[-1], lw=2, label='dead only')
axes1[1, 2].plot(z, PDI_tot[-1], lw=1.5, ls='--', label='combined')
axes1[1, 2].set_xlabel('z (m)'); axes1[1, 2].set_ylabel('PDI'); axes1[1, 2].set_title('PDI'); axes1[1, 2].legend(fontsize=7)

axes1[1, 3].plot(z, lamb0_s[-1], lw=2, label='λ₀')
axes1[1, 3].plot(z, mu0_s[-1], lw=2, ls='--', label='μ₀')
axes1[1, 3].set_xlabel('z (m)'); axes1[1, 3].set_ylabel('mol/m³'); axes1[1, 3].set_title('λ₀ vs μ₀ (QSSA check)'); axes1[1, 3].legend(fontsize=7)

plt.tight_layout()
plt.savefig('ldpe_pfr_v02_steady.png', dpi=150)

# %%
# ---- Figure 2: Dynamic evolution ------------------------------------------
# %%
z_idx    = [0, N//5, N//2, 4*N//5, N-1]
z_labels = [f'z={z[i]:.0f}m' for i in z_idx]
colors   = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle(f'Dynamic evolution  (N={N})', fontsize=11)
for idx, lbl, col in zip(z_idx, z_labels, colors):
    axes2[0].plot(sol.t, T_s[:,  idx]-273.15, color=col, label=lbl)
    axes2[1].plot(sol.t, X_s[:,  idx]*100,    color=col, label=lbl)
    axes2[2].plot(sol.t, Tc_s[:, idx]-273.15, color=col, label=lbl)
for ax, ttl, yl in zip(axes2,
    ['Reactor T(t)', 'Conversion(t)', 'Coolant Tc(t)'],
    ['T (°C)', 'Conv. (%)', 'Tc (°C)']):
    ax.set_xlabel('t (s)'); ax.set_ylabel(yl); ax.set_title(ttl); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('ldpe_pfr_v02_dynamic.png', dpi=150)

# %%
# ---- Figure 3: 2-D colormaps ----------------------------------------------
# %%
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4))
fig3.suptitle(f'Spatio-temporal maps  (N={N})', fontsize=11)
im0 = axes3[0].contourf(z, sol.t, T_s-273.15,  levels=40, cmap='hot')
fig3.colorbar(im0, ax=axes3[0], label='T (°C)'); axes3[0].set(xlabel='z (m)', ylabel='t (s)', title='T(z,t)')
im1 = axes3[1].contourf(z, sol.t, Tc_s-273.15, levels=40, cmap='cool_r')
fig3.colorbar(im1, ax=axes3[1], label='Tc (°C)'); axes3[1].set(xlabel='z (m)', ylabel='t (s)', title='Tc(z,t)')
im2 = axes3[2].contourf(z, sol.t, X_s*100,     levels=40, cmap='viridis')
fig3.colorbar(im2, ax=axes3[2], label='X (%)'); axes3[2].set(xlabel='z (m)', ylabel='t (s)', title='X(z,t)')
plt.tight_layout()
plt.savefig('ldpe_pfr_v02_spacetime.png', dpi=150)

plt.show()

# %%
# ---- Print summary --------------------------------------------------------
# %%
print('\n=== Steady-state summary at z=L ===')
print(f'  Conversion        : {X_s[-1,-1]*100:.2f} %')
print(f'  T_reactor (exit)  : {T_s[-1,-1]-273.15:.1f} °C')
print(f'  Tc (z=0, outlet)  : {Tc_s[-1,0]-273.15:.1f} °C')
print(f'  ΔT max (z)        : {(T_s[-1]-Tc_s[-1]).max():.1f} K')
print(f'  Mn (dead, exit)   : {Mn_dead[-1,-1]/1000:.3f} kg/mol')
print(f'  Mw (dead, exit)   : {Mw_dead[-1,-1]/1000:.3f} kg/mol')
print(f'  PDI (dead, exit)  : {PDI_dead[-1,-1]:.3f}')

# %%
# ---- Fix 8: Grid Convergence Test -----------------------------------------
# %%
def run_grid_convergence(N_list=(40, 100, 200), t_end_frac=2.0):
    """Run simulation at several N values and compare key output metrics."""
    results = {}
    for Ng in N_list:
        zg  = np.linspace(0.0, L, Ng)
        dzg = zg[1] - zg[0]
        NVg = NV
        jac = build_jac_sparsity(Ng, NVg)
        atol_g = np.tile(atol_per_var, Ng)
        ini_0g = ini_0; mono_0g = mono_0

        def odes_g(t, y):
            s = y.reshape(Ng, NVg)
            lamb0=s[:,0]; lamb1=s[:,1]; lamb2=s[:,2]
            mu0=s[:,3];   mu1=s[:,4];   mu2=s[:,5]
            ini=s[:,6];   mono=s[:,7];  T=s[:,8]; Tc=s[:,9]
            kd=arrhenius(A_kd,Ea_kd,T); kp=arrhenius(A_kp,Ea_kp,T)
            ktc=arrhenius(A_ktc,Ea_ktc,T); ktd=arrhenius(A_ktd,Ea_ktd,T)
            ktrm=arrhenius(A_ktrm,Ea_ktrm,T); ktrp=arrhenius(A_ktrp,Ea_ktrp,T)
            eps3=1e-12
            mu3_hk=np.where((mu0>eps3)&(mu1>eps3),
                             mu2*(2.0*mu0*mu2-mu1**2)/(mu0*mu1+eps3),0.0)
            mu3=np.maximum(mu3_hk, np.where(mu1>eps3, mu2**2/(mu1+eps3), 0.0))
            R_l0=2.0*f_eff*kd*ini-(ktc+ktd)*lamb0**2
            R_l1=(kp*mono*lamb0+ktrm*mono*(lamb0-lamb1)+ktrp*(lamb0*mu2-lamb1*mu1)-(ktc+ktd)*lamb0*lamb1)
            R_l2=(kp*mono*(2.0*lamb1+lamb0)+ktrm*mono*(lamb0-lamb2)+ktrp*(lamb0*mu3-lamb2*mu1)-(ktc+ktd)*lamb0*lamb2)
            R_m0=ktrm*mono*lamb0+(0.5*ktc+ktd)*lamb0**2
            R_m1=(ktrm*mono*lamb1+ktrp*(lamb1*mu1-lamb0*mu2)+(ktc+ktd)*lamb0*lamb1)
            R_m2=(ktrm*mono*lamb2+ktd*lamb0*lamb2+ktc*(lamb0*lamb2+lamb1**2)+ktrp*(lamb2*mu1-lamb0*mu3))
            R_ini=-kd*ini; R_mono=-kp*mono*lamb0
            R_T=(-dH_p)/(rho*Cp)*kp*mono*lamb0-h_r*(T-Tc)
            R_Tc=h_j*(T-Tc)
            dydt=np.zeros_like(s)
            for j,(C,R,BC) in enumerate(zip(
                [lamb0,lamb1,lamb2,mu0,mu1,mu2,ini,mono,T],
                [R_l0,R_l1,R_l2,R_m0,R_m1,R_m2,R_ini,R_mono,R_T],
                [0.,0.,0.,0.,0.,0.,ini_0g,mono_0g,T_0]
            )):
                C_up=np.empty(Ng); C_up[0]=BC; C_up[1:]=C[:-1]
                dCdz=(C-C_up)/dzg
                dydt[0,j]=0.; dydt[1:,j]=R[1:]-v*dCdz[1:]
            Tc_dn=np.empty(Ng); Tc_dn[:-1]=Tc[1:]; Tc_dn[-1]=Tc_in
            dTcdz=(Tc_dn-Tc)/dzg
            dydt[:Ng-1,9]=R_Tc[:Ng-1]+v_c*dTcdz[:Ng-1]; dydt[Ng-1,9]=0.
            return dydt.ravel()

        y0g=np.zeros((Ng,NVg))
        y0g[:,7]=mono_0g; y0g[:,8]=T_0; y0g[:,9]=Tc_in; y0g[0,6]=ini_0g; y0g[Ng-1,9]=Tc_in
        t_e=t_end_frac*tau_res
        s=solve_ivp(odes_g,(0.,t_e),y0g.ravel(),method='Radau',
                    rtol=1e-4,atol=atol_g,jac_sparsity=jac,
                    t_eval=[t_e],dense_output=False)
        Y_f=s.y.T.reshape(1,Ng,NVg)
        T_f=Y_f[0,:,8]; Tc_f=Y_f[0,:,9]
        mono_f=Y_f[0,:,7]; mu1_f=Y_f[0,:,4]; mu0_f=Y_f[0,:,3]
        X_exit  = float(1.0-mono_f[-1]/mono_0)
        T_max   = float(T_f.max()-273.15)
        Mn_exit = float(Mw_mono*mu1_f[-1]/(mu0_f[-1]+1e-30))
        results[Ng] = {'X_exit_%': X_exit*100, 'T_max_C': T_max, 'Mn_exit_g/mol': Mn_exit}
        print(f'  N={Ng:4d}  dz={dzg:6.2f}m  X={X_exit*100:.3f}%  T_max={T_max:.2f}°C  Mn={Mn_exit:.1f} g/mol')
    return results

print('\n--- Grid convergence study (t = 1τ, N=40 vs 100) ---')
gc = run_grid_convergence(N_list=[40, 100], t_end_frac=1.0)

# %%
# =============================================================================
# ANALYSIS OF ISSUES NOT FIXED (informational only)
# =============================================================================
#
# Issue 2 — Constant density ρ = 600 kg/m³
#   In reality, at 2000 bar and temperatures spanning 150–500 °C, the mixture
#   density of supercritical ethylene + growing LDPE chains varies significantly:
#   • At T=150 °C, ρ_C2H4 ≈ 550–650 kg/m³ (NIST, 2000 bar)
#   • At T=300 °C, ρ_C2H4 ≈ 400–500 kg/m³  → ~20–30 % lower
#   • Polymer fraction (ρ_LDPE ≈ 920 kg/m³) partially offsets the drop
#   Consequence: as T rises down the reactor, the fixed-v assumption fails
#   (continuity: ρ·v = const → v increases as ρ drops).  A proper model
#   needs ρ = f(T, X) from an EoS (e.g., SL-EoS or PC-SAFT) and a separate
#   continuity equation for the axial velocity profile.
#
# Issue 3 — Pressure drop and pressure-dependent Arrhenius
#   LDPE tubular reactors experience ΔP ~ 200–500 bar over 1.5–2 km.
#   Two effects are omitted:
#   (a) Ergun/Darcy friction pressure drop changes local operating pressure,
#       affecting density and Arrhenius pre-exponential factors.
#   (b) k_p and k_tc have measurable activation volumes in supercritical
#       ethylene:  k_p(P,T) = k_p(P_ref,T) · exp(−ΔV‡_p(P−P_ref)/RT)
#       Typical ΔV‡_p ≈ −27 cm³/mol → k_p roughly doubles from 2000→1000 bar.
#   Implementing this requires: Fanning friction factor → dP/dz ODE,
#   and Arrhenius corrected to: k = A·exp(−Ea/RT)·exp(−ΔV‡·P/RT).
#
# Issue 7 — LCB term in μ₀ equation
#   The docx specification (v3_fixed, section 2-B) explicitly states:
#   "LCB has zero net effect on chain count → no k_trp term in dμ₀/dt".
#   This is because a chain-transfer-to-polymer event converts one dead chain
#   of length n to one live radical of length n plus one new dead chain growing
#   from a branch point — net change in dead chain count is zero.
#   Code is consistent with the document and intentional.
# =============================================================================
