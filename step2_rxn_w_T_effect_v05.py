# %%
# Importing Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import time

# =============================================================================
# LDPE Free-Radical Polymerisation — Dynamic 1D PFR Model  v05
#
# Key changes from v04:
#   v05-1  Chain transfer to CTA (chain-transfer agent) added.
#          CTA concentration is a new state variable (index 10 in the state
#          vector). F_CTA (CTA feed flow rate) is the primary manipulated
#          variable for Mn control in the RL design.
#
#          Mechanism (QSSA on CTA radical; re-initiation is instantaneous):
#            Pₙ + CTA  →  Dₙ  +  P₁         rate = k_tr,CTA · [CTA] · λ₀
#
#          Moment contributions:
#            dλ₀/dt|_CTA = 0                        (one Pₙ removed, one P₁ added)
#            dλ₁/dt|_CTA = −k_tr,CTA·[CTA]·λ₁ + k_tr,CTA·[CTA]·λ₀
#                          (net ≈ −k_tr,CTA·[CTA]·λ₁  since λ₁ >> λ₀)
#            dλ₂/dt|_CTA = −k_tr,CTA·[CTA]·λ₂ + k_tr,CTA·[CTA]·λ₀
#                          (λ₂ decreasing, new P₁ contributes 1²·λ₀)
#            dμ₀/dt|_CTA = +k_tr,CTA·[CTA]·λ₀      (new dead chain per event)
#            dμ₁/dt|_CTA = +k_tr,CTA·[CTA]·λ₁      (dead chain carries full length)
#            dμ₂/dt|_CTA = +k_tr,CTA·[CTA]·λ₂
#
#          CTA mass balance in the PFR:
#            d[CTA]/dt = −v · d[CTA]/dz  − k_tr,CTA · [CTA] · λ₀
#            BC: [CTA](z=0) = cta_0   (proportional to F_CTA)
#            Relationship: cta_0 [mol/m³] = F_CTA [m³/s] · [CTA]_feed [mol/m³]
#                                           / (v · A_r)  where A_r = π/4·D²
#
#          Parameters (mercaptan-type CTA, e.g. dodecyl mercaptan):
#            A_ktr_cta  = 1.0e5  m³/(mol·s)
#            Ea_ktr_cta = 32 000  J/mol
#            → chain-transfer constant Cs = k_tr,CTA/kp ≈ 0.8 at 195 °C
#
#          Physical motivation:
#            CTA chain transfer terminates growing chains and re-initiates short
#            P₁ radicals, reducing both Mn and Mw without affecting conversion
#            strongly. It is the primary lever for Mn control in the RL design.
#
#   v05-2  State vector extended from 10 to 11 variables (NV = 11):
#            [λ₀, λ₁, λ₂, μ₀, μ₁, μ₂, ini, mono, T, Tc, CTA]
#            indices: 0   1   2   3   4   5   6    7    8  9   10
#
# Inherited from v04:
#   v04-1  β-scission of living radicals (lumped effective, no MCR species).
#   v04-2  β-scission effect study: base case compared with β-scission disabled.
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
A_ktrm = 1550.;    Ea_ktrm =  47_000.0   # chain transfer to monomer
A_ktrp = 3.0e-1;   Ea_ktrp =  50_000.0   # chain transfer to polymer (LCB)
A_kbs  = 1.0e11;   Ea_kbs  = 130_000.0   # β-scission (v04-1)
# v05-1: CTA chain transfer (mercaptan-type; Cs ≈ 0.8 at 195 °C)
A_ktr_cta  = 1.0e5;  Ea_ktr_cta = 32_000.0

def arrhenius(A, Ea, T):
    return A * np.exp(-Ea / (R_gas * T))

# %%
# ---- Spatial Grid ----------------------------------------------------------
N  = 200
NV = 11          # v05-1: +1 for [CTA] (index 10)
z  = np.linspace(0.0, L, N)
dz = z[1] - z[0]
print(f'Grid: N={N}, NV={NV}, dz={dz:.2f} m, τ_res={L/v:.1f} s')

# %%
# ---- Jacobian Sparsity -----------------------------------------------------
def build_jac_sparsity(N, NV=11):
    """
    Block-banded sparsity for NV=11 state vector per node.
    Index  0-8 : λ₀,λ₁,λ₂,μ₀,μ₁,μ₂,ini,mono,T — upwind (forward flow)
    Index  9   : Tc  — downwind (counter-current coolant)
    Index 10   : CTA — upwind (forward flow, same as monomer)
    """
    rows, cols = [], []
    for i in range(N):
        for j in range(NV):
            row = i * NV + j
            # Full within-node coupling (all reactions couple all state vars)
            for k in range(NV):
                rows.append(row); cols.append(i * NV + k)
            # Upwind coupling for forward-flowing species (j=0..8 and j=10)
            if (j < 9 or j == 10) and i > 0:
                rows.append(row); cols.append((i - 1) * NV + j)
            # Downwind coupling for counter-current coolant (j=9)
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
cta_0  = 10.0          # mol/m³  default inlet CTA concentration
                        # (set by F_CTA; Cs·[CTA]/[M] ≈ 4e-4 ≈ 1.5·Cm_monomer)

# %%
# ---- Tolerance Array -------------------------------------------------------
# 11 tolerances: one per state variable in order [λ₀,λ₁,λ₂,μ₀,μ₁,μ₂,ini,mono,T,Tc,CTA]
atol_per_var = np.array([1e-14, 1e-11, 1e-7, 1e-11, 1e-8, 1e-3, 1e-8, 1e-1, 1e-4, 1e-4, 1e-6])
atol_vec     = np.tile(atol_per_var, N)

# %%
# ---- ODE Builder -----------------------------------------------------------
def make_odes(ini_0_, mono_0_, T_0_, Tc_in_, h_r_, h_j_, cta_0_,
              kp_factor=1.0, N_=None, enable_bs=True):
    """
    Return ODE function for given operating conditions.
    cta_0_    : inlet CTA concentration [mol/m³] — controlled via F_CTA
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
        cta  =s[:,10]                               # v05-1: [CTA] state variable

        kd       = arrhenius(A_kd,      Ea_kd,      T)
        kp       = arrhenius(A_kp,      Ea_kp,      T) * kp_factor
        ktc      = arrhenius(A_ktc,     Ea_ktc,     T)
        ktd      = arrhenius(A_ktd,     Ea_ktd,     T)
        ktrm     = arrhenius(A_ktrm,    Ea_ktrm,    T)
        ktrp     = arrhenius(A_ktrp,    Ea_ktrp,    T)
        kbs      = arrhenius(A_kbs,     Ea_kbs,     T) if enable_bs else np.zeros_like(T)
        ktr_cta  = arrhenius(A_ktr_cta, Ea_ktr_cta, T)  # v05-1

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
        #
        # v05-1 CTA chain transfer: Pₙ + CTA → Dₙ + P₁  (rate = ktr_cta·[CTA]·λ₀)
        #   dλ₀|_CTA = 0          (conserved: Pₙ removed, P₁ added)
        #   dλ₁|_CTA = −ktr_cta·[CTA]·λ₁ + ktr_cta·[CTA]·λ₀
        #   dλ₂|_CTA = −ktr_cta·[CTA]·λ₂ + ktr_cta·[CTA]·λ₀  (λ₂ decreasing)
        R_l0 = 2.*f_eff*kd*ini - (ktc+ktd)*lamb0**2
        R_l1 = (kp*mono*lamb0 + ktrm*mono*(lamb0-lamb1)
                + ktrp*(lamb0*mu2-lamb1*mu1) - (ktc+ktd)*lamb0*lamb1
                - kbs*lamb1/2.
                + ktr_cta*cta*(lamb0 - lamb1))          # v05-1 CTA terms
        R_l2 = (kp*mono*(2.*lamb1+lamb0) + ktrm*mono*(lamb0-lamb2)
                + ktrp*(lamb0*mu3-lamb2*mu1) - (ktc+ktd)*lamb0*lamb2
                - kbs*lamb2*2./3.
                + ktr_cta*cta*(lamb0 - lamb2))          # v05-1 CTA terms

        # ---- Dead chain moments ---------------------------------------------
        # β-scission dead fragment D_{n/2}:
        #   dμ₀|_bs = +k_bs·λ₀,  dμ₁|_bs = +k_bs·λ₁/2,  dμ₂|_bs = +k_bs·λ₂/3
        #
        # v05-1 CTA transfer dead chain Dₙ:
        #   dμ₀|_CTA = +ktr_cta·[CTA]·λ₀
        #   dμ₁|_CTA = +ktr_cta·[CTA]·λ₁
        #   dμ₂|_CTA = +ktr_cta·[CTA]·λ₂
        R_m0 = (ktrm*mono*lamb0 + (0.5*ktc+ktd)*lamb0**2
                + kbs*lamb0
                + ktr_cta*cta*lamb0)                    # v05-1
        R_m1 = (ktrm*mono*lamb1 + ktrp*(lamb1*mu1-lamb0*mu2)
                + (ktc+ktd)*lamb0*lamb1
                + kbs*lamb1/2.
                + ktr_cta*cta*lamb1)                    # v05-1
        R_m2 = (ktrm*mono*lamb2 + ktd*lamb0*lamb2
                + ktc*(lamb0*lamb2+lamb1**2) + ktrp*(lamb2*mu1-lamb0*mu3)
                + kbs*lamb2/3.
                + ktr_cta*cta*lamb2)                    # v05-1

        R_ini  = -kd * ini
        R_mono = -kp * mono * lamb0
        RT     = (-dH_p)/(rho*Cp)*kp*mono*lamb0 - h_r_*(T - Tc)
        RT    -= np.maximum(0., T - T_safe) * 10.0

        # v05-1: CTA mass balance — consumed by chain transfer
        # d[CTA]/dt = −v·d[CTA]/dz − ktr_cta·[CTA]·λ₀
        # BC: [CTA](z=0) = cta_0_  (proportional to F_CTA)
        R_cta = -ktr_cta * cta * lamb0

        R_Tc = h_j_ * (T - Tc)

        dydt = np.zeros_like(s)
        # Forward-flowing variables (upwind differencing, indices 0-8)
        for j, (C, R, BC) in enumerate(zip(
            [lamb0, lamb1, lamb2, mu0,  mu1,  mu2,  ini,    mono,   T  ],
            [R_l0,  R_l1,  R_l2,  R_m0, R_m1, R_m2, R_ini,  R_mono, RT ],
            [0.,    0.,    0.,    0.,   0.,   0.,   ini_0_, mono_0_, T_0_]
        )):
            C_up     = np.empty(N_)
            C_up[0]  = BC
            C_up[1:] = C[:-1]
            dCdz     = (C - C_up) / dz_
            dydt[0, j]  = 0.
            dydt[1:, j] = R[1:] - v * dCdz[1:]

        # Counter-current coolant (downwind, index 9)
        Tc_dn      = np.empty(N_)
        Tc_dn[:-1] = Tc[1:]
        Tc_dn[-1]  = Tc_in_
        dTcdz      = (Tc_dn - Tc) / dz_
        dydt[:N_-1, 9] = R_Tc[:N_-1] + v_c * dTcdz[:N_-1]
        dydt[N_-1,  9] = 0.

        # v05-1: CTA (forward flow, upwind, index 10)
        cta_up     = np.empty(N_)
        cta_up[0]  = cta_0_
        cta_up[1:] = cta[:-1]
        dCTAdz     = (cta - cta_up) / dz_
        dydt[0,  10] = 0.
        dydt[1:, 10] = R_cta[1:] - v * dCTAdz[1:]

        return dydt.ravel()

    return pfr_odes

# %%
# ---- Run one simulation ----------------------------------------------------
def run_pfr(T0_C=195., ini_0_=0.01, Tc_C=130., U_=1500., P_bar=2000.,
            cta_0_=10.0,
            N_=200, rtol_=1e-4, t_frac=4., enable_bs=True):
    """
    Integrate LDPE PFR to pseudo-steady state.
    cta_0_ : inlet CTA concentration [mol/m³]; set by F_CTA.
    """
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
    y0[:, 7]  = mono_0
    y0[:, 8]  = T0_K
    y0[:, 9]  = Tc_K
    y0[:, 10] = cta_0_        # v05-1: initialise CTA profile at inlet value
    y0[0,  6] = ini_0_
    y0[N_-1, 9] = Tc_K

    odes = make_odes(ini_0_, mono_0, T0_K, Tc_K, h_r_, h_j_, cta_0_,
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
    cta_f = Y[-1,:,10]
    eps = 1e-30
    Mn_z  = Mw_mono * (mu1_f+eps) / (mu0_f+eps)
    Mw_z  = Mw_mono * (mu2_f+eps) / (mu1_f+eps)
    PDI_z = Mw_z / Mn_z
    X_z   = 1. - mono_f / mono_0

    return dict(
        ok        = sol.status == 0,
        t_cpu     = elapsed,
        Y         = Y,
        sol       = sol,
        z         = np.linspace(0., L, N_),
        X_exit    = float(X_z[-1]),
        T_peak    = float(T_f.max() - 273.15),
        Mn_exit   = float(Mn_z[-1]) / 1000.,
        Mw_exit   = float(Mw_z[-1]) / 1000.,
        PDI_exit  = float(PDI_z[-1]),
        CTA_exit  = float(cta_f[-1]),
        Mn_z      = Mn_z / 1000.,
        Mw_z      = Mw_z / 1000.,
        PDI_z     = PDI_z,
        X_z       = X_z,
        T_z       = T_f - 273.15,
        Tc_z      = Tc_f - 273.15,
        CTA_z     = cta_f,
    )

# %%
# ---- CTA chain-transfer constant check at key temperatures -----------------
print('\n--- CTA chain-transfer constant vs. temperature ---')
print(f'  A_ktr_cta = {A_ktr_cta:.2e}  m³/(mol·s)')
print(f'  Ea_ktr_cta = {Ea_ktr_cta:.0f}  J/mol')
print(f'\n{"T (°C)":>8}  {"ktr_cta":>12}  {"kp":>12}  {"Cs=ktr/kp":>10}')
print('-'*52)
for Tc_ in [175., 195., 210., 230., 250., 280.]:
    T_K  = Tc_ + 273.15
    kt_  = arrhenius(A_ktr_cta, Ea_ktr_cta, T_K)
    kp_  = arrhenius(A_kp,      Ea_kp,      T_K)
    Cs_  = kt_ / kp_
    print(f'{Tc_:8.1f}  {kt_:12.4e}  {kp_:12.4e}  {Cs_:10.4f}')

# %%
# ---- Mayo equation: CTA contribution to chain length ----------------------
print('\n--- Mayo equation at T₀=195 °C, cta_0 sweep ---')
T_K0 = 195. + 273.15
kp0  = arrhenius(A_kp,      Ea_kp,      T_K0)
kd0  = arrhenius(A_kd,      Ea_kd,      T_K0)
ktc0 = arrhenius(A_ktc,     Ea_ktc,     T_K0)
ktd0 = arrhenius(A_ktd,     Ea_ktd,     T_K0)
km0  = arrhenius(A_ktrm,    Ea_ktrm,    T_K0)
kct0 = arrhenius(A_ktr_cta, Ea_ktr_cta, T_K0)
l0_0 = np.sqrt(2.*f_eff*kd0*ini_0 / (ktc0+ktd0))
print(f'  kp         = {kp0:.4e}  ktrm = {km0:.4e}  Cm = {km0/kp0:.4e}')
print(f'  ktr_cta    = {kct0:.4e}  Cs   = {kct0/kp0:.4f}')
print(f'  λ₀ (QSSA)  = {l0_0:.4e} mol/m³')
print(f'  {"cta_0":>10}  {"1/DPn_CTA":>12}  {"DPn":>10}  {"Mn (kg/mol)":>13}')
print(f'  {"-"*50}')
for c0 in [0., 2., 5., 10., 20., 50.]:
    inv_DPn = (km0*mono_0 + kct0*c0 + 0.5*(ktc0+ktd0)*l0_0) / (kp0*mono_0)
    DPn_    = 1./inv_DPn
    Mn_     = DPn_ * Mw_mono / 1000.
    print(f'  {c0:10.1f}  {inv_DPn:12.4e}  {DPn_:10.0f}  {Mn_:13.2f}')

# %%
# ---- Base-case run (with CTA, with β-scission) -----------------------------
print('\n--- Base case  (T₀=195 °C, ini₀=0.01, Tc=130 °C, U=1500, cta₀=10 mol/m³) ---')
base = run_pfr()
print(f'  CPU time    : {base["t_cpu"]:.1f} s   solver ok={base["ok"]}')
print(f'  X (exit)    : {base["X_exit"]*100:.2f} %       [5–15 %]  {"✓" if 0.05<=base["X_exit"]<=0.15 else "✗"}')
print(f'  T peak      : {base["T_peak"]:.1f} °C    [200–300 °C]  {"✓" if 200<=base["T_peak"]<=300 else "✗"}')
print(f'  Mn (exit)   : {base["Mn_exit"]:.2f} kg/mol  [30–150]  {"✓" if 30<=base["Mn_exit"]<=150 else "✗"}')
print(f'  Mw (exit)   : {base["Mw_exit"]:.2f} kg/mol  [100–500] {"✓" if 100<=base["Mw_exit"]<=500 else "✗"}')
print(f'  PDI         : {base["PDI_exit"]:.2f}          [3–15]    {"✓" if 3<=base["PDI_exit"]<=15 else "✗"}')
print(f'  CTA (exit)  : {base["CTA_exit"]:.4f} mol/m³ (inlet: {cta_0:.1f})')

# %%
# ---- CTA sensitivity: Mn vs. cta_0 ----------------------------------------
cta_cases = [0., 2., 5., 10., 20., 40.]
print('\n--- CTA sensitivity  (N=100, T₀=195 °C, ini₀=0.01, Tc=130 °C, U=1500) ---')
print(f'{"cta₀":>8}  {"T_pk":>6}  {"X%":>5}  {"Mn":>8}  {"Mw":>8}  {"PDI":>6}  {"CTA_exit":>10}  ok')
print('-'*72)
cta_sens = {}
for c0 in cta_cases:
    r = run_pfr(cta_0_=c0, N_=100, rtol_=1e-3, t_frac=3.)
    cta_sens[c0] = r
    print(f'{c0:8.1f}  {r["T_peak"]:6.1f}  {r["X_exit"]*100:5.2f}  '
          f'{r["Mn_exit"]:8.2f}  {r["Mw_exit"]:8.2f}  {r["PDI_exit"]:6.2f}  '
          f'{r["CTA_exit"]:10.4f}  {"ok" if r["ok"] else "FAIL"}')

# %%
# ---- Temperature sensitivity with CTA (base cta_0=10) vs. no CTA ----------
T0_cases = [180., 185., 190., 195., 200., 205., 210.]
print('\n--- Temperature sensitivity  (N=100, cta₀=10 vs. cta₀=0) ---')
print(f'{"T₀":>5}  {"T_pk":>6}  {"X%":>5}  '
      f'{"Mn_cta":>8} {"Mn_no":>8}  {"ΔMn%":>7}  '
      f'{"Mw_cta":>8} {"PDI_cta":>8}  ok')
print('-'*80)
T_sens_cta  = {}
T_sens_nocta = {}
for T0c in T0_cases:
    r_cta  = run_pfr(T0_C=T0c, cta_0_=10.0, N_=100, rtol_=1e-3, t_frac=3.)
    r_no   = run_pfr(T0_C=T0c, cta_0_=0.0,  N_=100, rtol_=1e-3, t_frac=3.)
    T_sens_cta[T0c]   = r_cta
    T_sens_nocta[T0c] = r_no
    dMn = (r_no['Mn_exit'] - r_cta['Mn_exit']) / max(r_no['Mn_exit'], 1.) * 100.
    print(f'{T0c:5.1f}  {r_cta["T_peak"]:6.1f}  {r_cta["X_exit"]*100:5.2f}  '
          f'{r_cta["Mn_exit"]:8.2f} {r_no["Mn_exit"]:8.2f}  {dMn:7.2f}%  '
          f'{r_cta["Mw_exit"]:8.2f} {r_cta["PDI_exit"]:8.2f}  '
          f'{"ok" if r_cta["ok"] else "FAIL"}')

# %%
# ---- Figure 1: Base-case steady-state profiles (with CTA) ------------------
fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
fig1.suptitle(f'LDPE PFR v05 — Base-case steady state  '
              f'(T₀=195 °C, cta₀=10 mol/m³, β-scission ON)', fontsize=10)
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

# v05-1: CTA concentration profile
axes[1,1].plot(zb, base['CTA_z'], 'm', lw=2)
axes[1,1].axhline(cta_0, color='gray', ls=':', lw=0.8, label=f'inlet {cta_0} mol/m³')
axes[1,1].set(xlabel='z (m)', ylabel='[CTA] (mol/m³)', title='CTA profile')
axes[1,1].legend(fontsize=8)

Y   = base['Y']
t_arr = base['sol'].t
T_dyn = Y[:,:,8] - 273.15
X_dyn = 1. - Y[:,:,7] / mono_0

axes[1,2].contourf(zb, t_arr, T_dyn, levels=40, cmap='hot')
axes[1,2].set(xlabel='z (m)', ylabel='t (s)', title='T(z,t) map')

axes[1,3].plot(zb, base['Y'][-1,:,6], 'm', lw=2)
axes[1,3].set(xlabel='z (m)', ylabel='[I] (mol/m³)', title='Initiator profile')

plt.tight_layout()
plt.savefig('ldpe_pfr_v05_basecase.png', dpi=150)
print('\nFigure saved: ldpe_pfr_v05_basecase.png')

# %%
# ---- Figure 2: CTA sensitivity — Mn, Mw, PDI vs. cta_0 --------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle('v05 — CTA effect on Mn, Mw, PDI  (T₀=195 °C, N=100)', fontsize=11)

cta_arr  = np.array(cta_cases)
Mn_arr   = np.array([cta_sens[c]['Mn_exit']  for c in cta_cases])
Mw_arr   = np.array([cta_sens[c]['Mw_exit']  for c in cta_cases])
PDI_arr  = np.array([cta_sens[c]['PDI_exit'] for c in cta_cases])
X_arr    = np.array([cta_sens[c]['X_exit']   for c in cta_cases])

ax = axes2[0]
ax.plot(cta_arr, Mn_arr,  'o-', lw=2, color='royalblue')
ax.axhline(30,  color='gray', ls=':', lw=0.8)
ax.axhline(150, color='gray', ls=':', lw=0.8)
ax.set(xlabel='cta₀ (mol/m³)', ylabel='Mn (kg/mol)',
       title='Mn vs. inlet CTA\n(primary Mn control lever)')

ax = axes2[1]
ax.plot(cta_arr, Mw_arr,  's-', lw=2, color='tomato',    label='Mw')
ax.plot(cta_arr, Mn_arr,  'o--',lw=2, color='royalblue', label='Mn')
ax.axhline(100, color='gray', ls=':', lw=0.8)
ax.axhline(500, color='gray', ls=':', lw=0.8)
ax.set(xlabel='cta₀ (mol/m³)', ylabel='Molar mass (kg/mol)', title='Mn & Mw vs. CTA')
ax.legend(fontsize=8)

ax = axes2[2]
ax2b = ax.twinx()
ax.plot(cta_arr, PDI_arr, '^-', lw=2, color='seagreen', label='PDI')
ax2b.plot(cta_arr, X_arr*100, 'd--', lw=2, color='darkorange', label='X (%)')
ax.set(xlabel='cta₀ (mol/m³)', ylabel='PDI', title='PDI & X vs. CTA')
ax2b.set_ylabel('X (%)', color='darkorange')
ax2b.tick_params(axis='y', colors='darkorange')
ax.legend(loc='upper left', fontsize=8)
ax2b.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('ldpe_pfr_v05_cta_sensitivity.png', dpi=150)
print('Figure saved: ldpe_pfr_v05_cta_sensitivity.png')

# %%
# ---- Figure 3: CTA spatial profiles for different cta_0 --------------------
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))
fig3.suptitle('v05 — CTA spatial profiles (T₀=195 °C, N=100)', fontsize=11)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cta_cases)))

for (c0, r), col in zip(cta_sens.items(), colors):
    lbl = f'cta₀={c0:.0f}'
    zr = r['z']
    axes3[0].plot(zr, r['CTA_z'],    color=col, label=lbl, lw=1.5)
    axes3[1].plot(zr, r['Mn_z'],     color=col, label=lbl, lw=1.5)
    axes3[2].plot(zr, r['Mw_z'],     color=col, label=lbl, lw=1.5)
    axes3[3].plot(zr, r['PDI_z'],    color=col, label=lbl, lw=1.5)

for ax, yl, ttl in zip(axes3,
    ['[CTA] (mol/m³)', 'Mn (kg/mol)', 'Mw (kg/mol)', 'PDI'],
    ['CTA concentration', 'Number-avg MW', 'Weight-avg MW', 'PDI']):
    ax.set(xlabel='z (m)', ylabel=yl, title=ttl)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('ldpe_pfr_v05_cta_spatial.png', dpi=150)
print('Figure saved: ldpe_pfr_v05_cta_spatial.png')

# %%
# ---- Figure 4: T sensitivity with CTA ON (cta₀=10) vs. no CTA -------------
fig4, axes4 = plt.subplots(1, 4, figsize=(16, 4))
fig4.suptitle('v05 — Temperature sensitivity  (cta₀=10 vs. cta₀=0)', fontsize=11)
colors2 = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(T0_cases)))

for (T0c, r), col in zip(T_sens_cta.items(), colors2):
    lbl = f'{T0c:.0f} °C'
    zr = r['z']
    axes4[0].plot(zr, r['T_z'],     color=col, label=lbl, lw=1.5)
    axes4[1].plot(zr, r['X_z']*100, color=col, label=lbl, lw=1.5)
    axes4[2].plot(zr, r['Mn_z'],    color=col, label=lbl, lw=1.5)
    axes4[3].plot(zr, r['PDI_z'],   color=col, label=lbl, lw=1.5)

for ax, yl, ttl, (lo, hi) in zip(axes4,
    ['T (°C)', 'X (%)', 'Mn (kg/mol)', 'PDI'],
    ['Reactor temperature', 'Conversion', 'Number-avg MW', 'PDI'],
    [(200,300), (5,15), (30,150), (3,15)]):
    ax.axhline(lo, color='gray', ls=':', lw=0.8)
    ax.axhline(hi, color='gray', ls=':', lw=0.8)
    ax.set(xlabel='z (m)', ylabel=yl, title=ttl)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('ldpe_pfr_v05_T_sensitivity.png', dpi=150)
print('Figure saved: ldpe_pfr_v05_T_sensitivity.png')

# %%
# ---- QSSA analytical check with CTA ----------------------------------------
print('\n--- QSSA + CTA check at T₀=195 °C ---')
T_K  = 195. + 273.15
kd_  = arrhenius(A_kd,      Ea_kd,      T_K)
kp_  = arrhenius(A_kp,      Ea_kp,      T_K)
ktc_ = arrhenius(A_ktc,     Ea_ktc,     T_K)
ktd_ = arrhenius(A_ktd,     Ea_ktd,     T_K)
ktrm_= arrhenius(A_ktrm,    Ea_ktrm,    T_K)
kbs_ = arrhenius(A_kbs,     Ea_kbs,     T_K)
kct_ = arrhenius(A_ktr_cta, Ea_ktr_cta, T_K)
l0_  = np.sqrt(2.*f_eff*kd_*ini_0 / (ktc_+ktd_))
Cm_  = ktrm_/kp_
Cs_  = kct_/kp_
tau_ = L / v

print(f'  kd        = {kd_:.4e} s⁻¹   (t½ = {np.log(2)/kd_:.1f} s)')
print(f'  kp        = {kp_:.4f} m³/(mol·s)')
print(f'  ktrm      = {ktrm_:.4e}   Cm = {Cm_:.4e}')
print(f'  ktr_cta   = {kct_:.4e}   Cs = {Cs_:.4f}')
print(f'  kbs       = {kbs_:.4e} s⁻¹')
print(f'  λ₀        = {l0_:.4e} mol/m³')
print(f'\n  Mayo equation breakdown at cta₀=10 mol/m³:')
inv_kterm = ktrm_*mono_0 / (kp_*mono_0)
inv_kcta  = kct_*cta_0   / (kp_*mono_0)
inv_ktc   = 0.5*(ktc_+ktd_)*l0_ / (kp_*mono_0)
print(f'    1/DPn from ktrm : {inv_kterm:.4e}  ({inv_kterm/(inv_kterm+inv_kcta+inv_ktc)*100:.1f}%)')
print(f'    1/DPn from CTA  : {inv_kcta:.4e}  ({inv_kcta/(inv_kterm+inv_kcta+inv_ktc)*100:.1f}%)')
print(f'    1/DPn from term : {inv_ktc:.4e}  ({inv_ktc/(inv_kterm+inv_kcta+inv_ktc)*100:.1f}%)')
DPn_tot = 1./(inv_kterm + inv_kcta + inv_ktc)
Mn_tot  = DPn_tot * Mw_mono / 1000.
print(f'    DPn_QSSA = {DPn_tot:.0f}   Mn_QSSA = {Mn_tot:.2f} kg/mol  (with CTA)')
DPn_no  = 1./(inv_kterm + inv_ktc)
Mn_no   = DPn_no * Mw_mono / 1000.
print(f'    DPn_QSSA = {DPn_no:.0f}   Mn_QSSA = {Mn_no:.2f} kg/mol  (no CTA)')
print(f'    Mn reduction by CTA: {(1.-Mn_tot/Mn_no)*100:.1f}%')

plt.show()

# %%
# =============================================================================
# Physical basis for v05 changes
# =============================================================================
#
# v05-1  CTA chain-transfer mechanism in LDPE:
#        A growing chain radical Pₙ abstracts the labile H (or Cl in chlorinated
#        CTAs) from the CTA molecule, terminating the growing chain as a dead
#        polymer Dₙ and producing a highly reactive CTA radical (CTA•).
#        Under QSSA (CTA• is short-lived), CTA• immediately reacts with monomer
#        to form a new primary radical P₁:
#
#          Pₙ  +  CTA  →  Dₙ  +  CTA•           rate = k_tr,CTA · [CTA] · [Pₙ]
#          CTA•  +  M  →  P₁                     QSSA: fast, rate-limiting ≈ above
#
#        Net reaction:  Pₙ + CTA + M  →  Dₙ + P₁      (lumped, QSSA on CTA•)
#
#        Moment equations (method of moments):
#          dλ₀/dt|CTA = 0          Σn⁰(-[Pₙ]+δₙ₁) = 0  (radical count conserved)
#          dλ₁/dt|CTA = Σn¹(-[Pₙ]+δₙ₁)·k_tr·[CTA]
#                      = k_tr·[CTA]·(1·λ₀ - λ₁)
#                      = k_tr·[CTA]·(λ₀ - λ₁)
#          dλ₂/dt|CTA = k_tr·[CTA]·(1²·λ₀ - λ₂)
#                      = k_tr·[CTA]·(λ₀ - λ₂)
#          dμ₀/dt|CTA = +k_tr·[CTA]·λ₀           (new dead chain Dₙ per event)
#          dμ₁/dt|CTA = +k_tr·[CTA]·λ₁           (Dₙ carries full length n)
#          dμ₂/dt|CTA = +k_tr·[CTA]·λ₂
#
#        CTA mass balance (PFR, forward-flowing, consumed by chain transfer):
#          ∂[CTA]/∂t + v · ∂[CTA]/∂z = −k_tr,CTA · [CTA] · λ₀
#          BC:  [CTA](z=0, t) = cta_0   (set by F_CTA, the manipulated variable)
#
#        Link F_CTA → cta_0:
#          If CTA is injected as a pure stream (flow rate F_CTA [m³/s]) into the
#          main ethylene flow Q_main = v · A_r = 12 · (π/4·0.05²) ≈ 0.0236 m³/s:
#            cta_0 = F_CTA · [CTA]_feed / Q_main
#          Here [CTA]_feed is the CTA concentration in the injection stream.
#          For control design, cta_0 is used directly as the action variable,
#          proportional to F_CTA.
#
#        Parameters (mercaptan-type CTA, representative of dodecyl mercaptan):
#          A_ktr_cta  = 1.0e5  m³/(mol·s)
#          Ea_ktr_cta = 32 000  J/mol
#          → Cs = k_tr,CTA / kp ≈ 0.8 at 195 °C  (cf. lit. Cs = 0.3–2 for C12-SH)
#
#        RL design implication:
#          Increasing cta_0 (↑ F_CTA):
#            • reduces Mn (shorter chains, lower average MW) ✓
#            • slightly reduces Mw (dead chains from CTA transfer are shorter)
#            • PDI may decrease slightly (narrower chain-length distribution)
#            • X barely affected (CTA transfer does not consume monomer)
#          CTA is therefore the targeted Mn control lever in the RL design.
#
#        Literature:
#          Ehrlich & Mortimer, Adv. Polym. Sci. 7, 386–448 (1970)
#          Pladis & Kiparissides, AIChE J. 44(9), 2024–2039 (1998)
#          Kiparissides et al., Macromol. React. Eng. 4, 271–275 (2010)
# =============================================================================
