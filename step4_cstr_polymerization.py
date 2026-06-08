# %%
# Importing Packages
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — prevents GUI hang when run as script
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# =============================================================================
# LDPE Free-Radical Polymerisation — Dynamic CSTR (Autoclave) Model
#
# Reference equations: "CSTR Mass & Energy Bal.pdf"
#   [2-1] Mass balance  — scalar ODEs for each species/moment
#   [2-2] Energy balance — reactor T and jacket Tc
#
# Reactor type : CSTR / Autoclave (perfectly mixed, single spatial node)
#
# Core reactions (same kinetics as PFR step2 files):
#   1. Initiation:          I  →ᵏᵈ 2R•,   R• + M →ᵏⁱ P₁•
#   2. Propagation:         Pₙ• + M  →ᵏᵖ  Pₙ₊₁•
#   3. Transfer→monomer:    Pₙ• + M  →ᵏᵗʳᵐ Dₙ + P₁•
#   4. Transfer→polymer:    Pₙ• + Dₘ →ᵏᵗʳᵖ Dₙ + Pₘ•  (LCB)
#   5. Termination (comb.): Pₙ• + Pₘ• →ᵏᵗᶜ Dₙ₊ₘ
#   6. Termination (disp.): Pₙ• + Pₘ• →ᵏᵗᵈ Dₙ + Dₘ
#   7. Transfer→CTA:        Pₙ• + CTA →ᵏᶜᵗᵃ Dₙ + R•_CTA
#   8. β-Scission (kβ):     Pₙ• →ᵏᵝ  Pₘ• + Dₙ₋ₘ
#      ► In CSTR formulation kβ is lumped into the combined effective
#        chain-transfer group  K_eff = ktrm·[M] + ktr_CTA·[CTA] + kβ
#        following the PDF derivation (chain-transfer analogy).
#
# CSTR governing ODE (per PDF §[2-1]):
#   d[Ci]/dt = (Fin·[Ci]in − F·[Ci])/V + Ri
#   With Fin = F, τ = V/F:
#   d[Ci]/dt = ([Ci]in − [Ci])/τ + Ri      …(*)
#
# State vector y[11]:
#   [λ₀, λ₁, λ₂, μ₀, μ₁, μ₂, ini, mono, CTA, T, Tc]
#    0    1    2   3    4    5   6    7     8   9  10
#
# Moment closure (μ₃):
#   Hulburt–Katz:    μ₃ ≈ μ₂(2μ₀μ₂ − μ₁²) / (μ₀μ₁)
#   Conservative:    μ₃ ≈ μ₂² / μ₁
#   Applied:         μ₃ = max(H–K, conservative)
#
# Differences vs PFR (step2):
#   • No spatial grid — 11 scalar ODEs only
#   • Convection replaced by (C_in − C)/τ flow terms
#   • kβ lumped with ktrm·[M] + ktr_CTA·[CTA] in K_eff (PDF formulation)
#   • Energy balance: dT/dt = (Tin−T)/τ + heat_gen − UA/(V·ρ·Cp)·(T−Tc)
#   • Jacket balance: dTc/dt = Fc/Vc·(Tc,in−Tc) + UA/(Vc·ρc·Cp_c)·(T−Tc)
# =============================================================================

# %%
# ── Physical / Process Constants ──────────────────────────────────────────
R_gas   = 8.3145        # J/(mol·K)
Mw_mono = 28.054        # g/mol  (ethylene)
f_eff   = 0.8           # initiator efficiency

rho     = 600_000.0     # g/m³   reaction mixture density
Cp      = 1.7           # J/(g·K)
dH_p    = -93_000.0     # J/mol  polymerisation enthalpy (negative = exothermic)

rho_c   = 900_000.0     # g/m³   coolant (water)
Cp_c    = 4.18          # J/(g·K)

# %%
# ── Arrhenius Parameters (synced with step2_rxn_w_T_effect_v05) ──────────
A_kd      = 3.15e15;  Ea_kd      = 155_000.0   # initiator decomp           [s⁻¹]
A_kp      = 6.58e4;   Ea_kp      =  29_500.0   # propagation                [m³/(mol·s)]
A_ktc     = 2.0e5;    Ea_ktc     =   5_000.0   # termination combination     [m³/(mol·s)]
A_ktd     = 2.0e5;    Ea_ktd     =   5_000.0   # termination disproportion.  [m³/(mol·s)]
A_ktrm    = 1550.;    Ea_ktrm    =  47_000.0   # transfer to monomer         [m³/(mol·s)]
A_ktrp    = 3.0e-1;   Ea_ktrp    =  50_000.0   # transfer to polymer (LCB)   [m³/(mol·s)]
A_kbs     = 1.0e11;   Ea_kbs     = 130_000.0   # β-scission  kβ              [s⁻¹]
A_ktr_cta = 1.0e5;    Ea_ktr_cta =  32_000.0   # transfer to CTA             [m³/(mol·s)]

def arrhenius(A, Ea, T):
    """Arrhenius equation.  T in Kelvin, Ea in J/mol."""
    return A * np.exp(-Ea / (R_gas * T))

# %%
# ── CSTR / Autoclave Parameters ───────────────────────────────────────────
V_cstr   = 0.5          # m³    reactor volume (autoclave)
UA_cstr  = 15_000.0     # W/K   total heat-transfer capacity (jacket + coils)
                         #        U≈600 W/(m²·K), A≈25 m² → UA=15 kW/K (internal coils)
V_c      = 0.05         # m³    jacket coolant volume
F_c      = 0.001        # m³/s  coolant flow rate

# Derived heat-transfer coefficients [s⁻¹]
alpha_r = UA_cstr / (V_cstr * rho * Cp)       # reactor side
alpha_c = UA_cstr / (V_c    * rho_c * Cp_c)   # jacket side
beta_c  = F_c    / V_c                         # jacket dilution

# %%
# ── Default Feed / Operating Conditions ───────────────────────────────────
mono_in  = 2.005e4     # mol/m³   monomer (ethylene) inlet concentration
ini_in   = 0.005       # mol/m³   initiator inlet (low — CSTR has poor heat removal)
cta_in   = 10.0        # mol/m³   CTA inlet
T_in_def = 200.0 + 273.15   # K  feed temperature
Tc_in_def= 130.0 + 273.15   # K  coolant inlet temperature
tau_def  = 60.0        # s    residence time  (τ = V/F)

print('=' * 65)
print('  LDPE CSTR (Autoclave) Simulator  —  step4')
print('=' * 65)
print(f'  V_cstr   = {V_cstr}  m³       UA_cstr  = {UA_cstr:.0f}  W/K')
print(f'  α_r      = {alpha_r:.5f}  s⁻¹    α_c      = {alpha_c:.5f}  s⁻¹')
print(f'  β_c      = {beta_c:.5f}  s⁻¹')
print(f'  mono_in  = {mono_in:.3e} mol/m³  ini_in = {ini_in}  cta_in = {cta_in}')

# %%
# ── Moment Closure (Hulburt–Katz + conservative) ─────────────────────────
def mu3_closure(mu0, mu1, mu2, eps=1e-12):
    """
    Approximate μ₃ from (μ₀, μ₁, μ₂) using two closures (PDF §7):
      H-K  : μ₃ ≈ μ₂(2μ₀μ₂ − μ₁²) / (μ₀μ₁)   [Hulburt & Katz, Gamma dist.]
      cons : μ₃ ≈ μ₂² / μ₁                       [conservative lower bound]
    Returns max of both to avoid underestimation.
    """
    valid = (mu0 > eps) and (mu1 > eps)
    hk = mu2*(2.*mu0*mu2 - mu1**2) / (mu0*mu1 + eps) if valid else 0.
    cs = mu2**2 / (mu1 + eps)                          if mu1 > eps else 0.
    return max(hk, cs, 0.)

# %%
# ── CSTR ODE System ───────────────────────────────────────────────────────
def cstr_odes(t, y, tau, T_in, Tc_in, ini_in_, mono_in_, cta_in_,
              UA=UA_cstr, Vc=V_c, Fc=F_c, V=V_cstr):
    """
    Right-hand sides of all 11 CSTR ODEs.

    State:  y = [λ₀, λ₁, λ₂, μ₀, μ₁, μ₂, ini, mono, CTA, T, Tc]

    Parameters
    ----------
    tau     : residence time [s]
    T_in    : feed temperature [K]
    Tc_in   : coolant inlet temperature [K]
    ini_in_ : initiator inlet concentration [mol/m³]
    mono_in_: monomer inlet concentration [mol/m³]
    cta_in_ : CTA inlet concentration [mol/m³]
    """
    l0, l1, l2 = y[0], y[1], y[2]       # live chain moments λ₀,λ₁,λ₂
    m0, m1, m2 = y[3], y[4], y[5]       # dead chain moments μ₀,μ₁,μ₂
    ini, mono, cta = y[6], y[7], y[8]   # [I], [M], [CTA]
    T,  Tc         = y[9], y[10]        # reactor T, jacket Tc

    # --- Rate constants (Arrhenius, updated every step) -------------------
    kd      = arrhenius(A_kd,      Ea_kd,      T)
    kp      = arrhenius(A_kp,      Ea_kp,      T)
    ktc     = arrhenius(A_ktc,     Ea_ktc,     T)
    ktd     = arrhenius(A_ktd,     Ea_ktd,     T)
    ktrm    = arrhenius(A_ktrm,    Ea_ktrm,    T)
    ktrp    = arrhenius(A_ktrp,    Ea_ktrp,    T)
    kbs     = arrhenius(A_kbs,     Ea_kbs,     T)   # kβ (β-scission)
    ktr_cta = arrhenius(A_ktr_cta, Ea_ktr_cta, T)

    # --- Moment closure: μ₃ ----------------------------------------------
    mu3 = mu3_closure(m0, m1, m2)

    # --- Effective combined chain-transfer rate (PDF [2-1] §2, §3) -------
    #   K_eff = ktrm·[M] + ktr_CTA·[CTA] + kβ
    #   Groups monomer transfer, CTA transfer, β-scission into one term.
    K_eff = ktrm * mono + ktr_cta * cta + kbs

    # --- Dilution rate ----------------------------------------------------
    dil = 1.0 / tau    # = F/V  [s⁻¹]

    # ==========================================================
    # LIVE CHAIN MOMENTS  (λ_k)   — PDF [2-1] §2
    # ==========================================================
    # λ₀: radical count conserved (CTA, β-scission do not change count)
    #   dλ₀/dt = −(F/V)λ₀ + 2f·kd·[I] − (ktc+ktd)·λ₀²
    Rl0 = (-dil*l0
            + 2.*f_eff*kd*ini
            - (ktc + ktd)*l0**2)

    # λ₁: chain-length weighted radical moment
    #   dλ₁/dt = −(F/V)λ₁ + kp·[M]·λ₀
    #            + K_eff·(λ₀−λ₁)           ← transfer terminates long chain,
    #            + ktrp·(λ₀·μ₂ − λ₁·μ₁)   ← re-initiates short P₁
    #            − (ktc+ktd)·λ₀·λ₁
    Rl1 = (-dil*l1
            + kp*mono*l0
            + K_eff*(l0 - l1)
            + ktrp*(l0*m2 - l1*m1)
            - (ktc + ktd)*l0*l1)

    # λ₂: second moment of live chains
    #   dλ₂/dt = −(F/V)λ₂ + kp·[M]·(2λ₁+λ₀)
    #            + K_eff·(λ₀−λ₂)
    #            + ktrp·(λ₀·μ₃ − λ₂·μ₁)
    #            − (ktc+ktd)·λ₀·λ₂
    Rl2 = (-dil*l2
            + kp*mono*(2.*l1 + l0)
            + K_eff*(l0 - l2)
            + ktrp*(l0*mu3 - l2*m1)
            - (ktc + ktd)*l0*l2)

    # ==========================================================
    # DEAD CHAIN MOMENTS  (μ_k)  — PDF [2-1] §3
    # ==========================================================
    # μ₀: number of dead chains  (no ktrp term — LCB conserves chain count)
    #   dμ₀/dt = −(F/V)μ₀ + K_eff·λ₀ + (0.5·ktc + ktd)·λ₀²
    Rm0 = (-dil*m0
            + K_eff*l0
            + (0.5*ktc + ktd)*l0**2)

    # μ₁: total mass of dead chains
    #   dμ₁/dt = −(F/V)μ₁ + K_eff·λ₁
    #            + ktrp·(λ₁·μ₁ − λ₀·μ₂)   ← LCB: dead chain attacked & becomes live
    #            + (ktc+ktd)·λ₀·λ₁
    Rm1 = (-dil*m1
            + K_eff*l1
            + ktrp*(l1*m1 - l0*m2)
            + (ktc + ktd)*l0*l1)

    # μ₂: second moment of dead chains  (governs PDI via ktrp LCB)
    #   dμ₂/dt = −(F/V)μ₂ + K_eff·λ₂
    #            + ktd·λ₀·λ₂ + ktc·(λ₀·λ₂ + λ₁²)
    #            + ktrp·(λ₂·μ₁ − λ₀·μ₃)
    Rm2 = (-dil*m2
            + K_eff*l2
            + ktd*l0*l2
            + ktc*(l0*l2 + l1**2)
            + ktrp*(l2*m1 - l0*mu3))

    # ==========================================================
    # SPECIES CONCENTRATIONS  — PDF [2-1] §1
    # ==========================================================
    # Initiator  d[I]/dt = ([I]in−[I])/τ − kd·[I]
    R_ini  = (ini_in_ - ini)/tau - kd*ini

    # Monomer    d[M]/dt = ([M]in−[M])/τ − kp·[M]·λ₀
    R_mono = (mono_in_ - mono)/tau - kp*mono*l0

    # CTA        d[CTA]/dt = ([CTA]in−[CTA])/τ − ktr_CTA·[CTA]·λ₀
    R_cta  = (cta_in_ - cta)/tau - ktr_cta*cta*l0

    # ==========================================================
    # ENERGY BALANCE  — PDF [2-2]
    # ==========================================================
    # Reactor temperature
    # dT/dt = (Tin−T)/τ + (−ΔHp)/(ρ·Cp)·kp·[M]·λ₀ − UA/(V·ρ·Cp)·(T−Tc)
    #         − K_c·max(0, T − T_sp)    [temperature controller, setpoint 240 °C]
    # Physical basis: industrial LDPE autoclaves are temperature-controlled;
    # this term models an ideal proportional controller that boosts cooling when
    # T exceeds the setpoint.  With K_c = 15 K/s/K the offset is < 1 °C.
    a_r  = UA / (V * rho * Cp)
    R_T  = ((T_in - T) / tau
            + (-dH_p)/(rho*Cp) * kp * mono * l0
            - a_r * (T - Tc))
    T_sp = 240. + 273.15            # temperature controller setpoint [K]
    R_T -= max(0., T - T_sp) * 15.0 # proportional controller (K_c = 15 s⁻¹)

    # Jacket temperature
    # dTc/dt = Fc/Vc·(Tc,in−Tc) + UA/(Vc·ρc·Cp_c)·(T−Tc)
    a_c  = UA / (Vc * rho_c * Cp_c)
    R_Tc = (Fc/Vc)*(Tc_in - Tc) + a_c*(T - Tc)

    return [Rl0, Rl1, Rl2, Rm0, Rm1, Rm2,
            R_ini, R_mono, R_cta, R_T, R_Tc]

# %%
# ── run_cstr: integrate to pseudo-steady state ────────────────────────────
def run_cstr(tau=tau_def, T_in_C=200., ini_in_=None, mono_in_=mono_in,
             cta_in_=cta_in, Tc_in_C=130., UA=UA_cstr,
             t_end_frac=8., rtol=1e-5, atol=None, y0=None):
    """
    Integrate CSTR ODEs to pseudo-steady state.

    Parameters
    ----------
    tau       : residence time [s]
    T_in_C    : feed temperature [°C]
    ini_in_   : initiator inlet [mol/m³]  (default: ini_in global)
    cta_in_   : CTA inlet [mol/m³]
    Tc_in_C   : coolant inlet [°C]
    UA        : heat-transfer capacity [W/K]
    t_end_frac: integration length = t_end_frac × τ
    y0        : initial state vector (default: cold start)

    Returns dict with SS values and profiles.
    """
    if ini_in_ is None:
        ini_in_ = ini_in
    T_in  = T_in_C  + 273.15
    Tc_in = Tc_in_C + 273.15
    t_end = t_end_frac * tau

    # Default cold start: reactor filled with fresh feed, T = T_in
    if y0 is None:
        y0 = [0., 0., 0., 0., 0., 0., ini_in_, mono_in_, cta_in_, T_in, Tc_in]

    atol_ = atol if atol is not None else [
        1e-14, 1e-11, 1e-7, 1e-11, 1e-8, 1e-3,   # λ, μ moments
        1e-8,  1e-1,  1e-6,                         # ini, mono, CTA
        1e-4,  1e-4                                  # T, Tc
    ]

    t_eval = np.linspace(0., t_end, 500)
    t0 = time.time()
    sol = solve_ivp(
        lambda t, y: cstr_odes(t, y, tau, T_in, Tc_in,
                               ini_in_, mono_in_, cta_in_, UA),
        (0., t_end), y0, method='Radau',
        t_eval=t_eval, rtol=rtol, atol=atol_, dense_output=False
    )
    elapsed = time.time() - t0

    y_ss = sol.y[:, -1]  # steady-state values (final time point)
    l0_ss, l1_ss, l2_ss = y_ss[0], y_ss[1], y_ss[2]
    m0_ss, m1_ss, m2_ss = y_ss[3], y_ss[4], y_ss[5]
    mono_ss = y_ss[7]
    T_ss    = y_ss[9]
    Tc_ss   = y_ss[10]

    eps = 1e-30
    Mn_ss  = Mw_mono * (m1_ss + eps) / (m0_ss + eps)
    Mw_ss  = Mw_mono * (m2_ss + eps) / (m1_ss + eps)
    PDI_ss = Mw_ss / Mn_ss
    X_ss   = 1. - mono_ss / mono_in_

    return dict(
        ok      = sol.status == 0,
        t_cpu   = elapsed,
        sol     = sol,
        t       = sol.t,
        Y       = sol.y.T,          # shape (nt, 11)
        y_ss    = y_ss,
        X_ss    = float(X_ss),
        T_ss_C  = float(T_ss - 273.15),
        Tc_ss_C = float(Tc_ss - 273.15),
        Mn_ss   = float(Mn_ss) / 1000.,   # kg/mol
        Mw_ss   = float(Mw_ss) / 1000.,
        PDI_ss  = float(PDI_ss),
        lam0_ss = float(l0_ss),
        tau     = tau,
    )

# %%
# ── QSSA Check: expected SS values ─────────────────────────────────────────
print('\n--- QSSA analytical check at T=220 °C, τ=60 s ---')
T_q = 220. + 273.15
kd_q    = arrhenius(A_kd,      Ea_kd,      T_q)
kp_q    = arrhenius(A_kp,      Ea_kp,      T_q)
ktc_q   = arrhenius(A_ktc,     Ea_ktc,     T_q)
ktd_q   = arrhenius(A_ktd,     Ea_ktd,     T_q)
ktrm_q  = arrhenius(A_ktrm,    Ea_ktrm,    T_q)
ktr_q   = arrhenius(A_ktr_cta, Ea_ktr_cta, T_q)
tau_q   = 60.
ini_ss_q= ini_in / (1. + kd_q*tau_q)
# QSSA on λ₀ in CSTR: -λ₀/τ + 2f·kd·ini_ss − (ktc+ktd)·λ₀² = 0
# Approximate: 2f·kd·ini_ss ≈ λ₀/τ + (ktc+ktd)·λ₀²  → solve quadratic
a_coef  = ktc_q + ktd_q
b_coef  = 1./tau_q
c_coef  = -2.*f_eff*kd_q*ini_ss_q
disc    = b_coef**2 - 4.*a_coef*c_coef
l0_q    = (-b_coef + np.sqrt(disc)) / (2.*a_coef)
Keff_q  = ktrm_q*mono_in + ktr_q*cta_in
DPn_q   = kp_q*mono_in / (Keff_q + (ktc_q+ktd_q)*l0_q + 1./tau_q)
Mn_q    = DPn_q * Mw_mono / 1000.
X_q     = kp_q * mono_in * l0_q * tau_q / (1. + kp_q*l0_q*tau_q)
print(f'  kd      = {kd_q:.4e} s⁻¹   ini_ss ≈ {ini_ss_q:.4e} mol/m³')
print(f'  λ₀_ss   = {l0_q:.4e} mol/m³   (QSSA)')
print(f'  kp·[M]·λ₀·τ = {kp_q*mono_in*l0_q*tau_q:.4f}  → X ≈ {X_q*100:.1f}%')
print(f'  DPn     = {DPn_q:.0f}   Mn_QSSA = {Mn_q:.2f} kg/mol')

# %%
# ── Base-case CSTR run ─────────────────────────────────────────────────────
print('\n--- Base case  (τ=60 s, T_in=200 °C, ini=0.005, cta=10, Tc=130 °C) ---')
base = run_cstr()
print(f'  CPU time  : {base["t_cpu"]:.2f} s   solver ok={base["ok"]}')
print(f'  X (SS)    : {base["X_ss"]*100:.2f} %      [10–30 %]  {"✓" if 0.10<=base["X_ss"]<=0.30 else "✗"}')
print(f'  T_ss      : {base["T_ss_C"]:.1f} °C   [200–280 °C]  {"✓" if 200<=base["T_ss_C"]<=280 else "✗"}')
print(f'  Tc_ss     : {base["Tc_ss_C"]:.1f} °C')
print(f'  Mn (SS)   : {base["Mn_ss"]:.2f} kg/mol  [20–100]  {"✓" if 20<=base["Mn_ss"]<=100 else "✗"}')
print(f'  Mw (SS)   : {base["Mw_ss"]:.2f} kg/mol  [50–400]  {"✓" if 50<=base["Mw_ss"]<=400 else "✗"}')
print(f'  PDI       : {base["PDI_ss"]:.2f}          [1.5–3.0] {"✓" if 1.5<=base["PDI_ss"]<=3.0 else "✗"}')
print(f'  λ₀        : {base["lam0_ss"]:.3e} mol/m³')

# %%
# ── Residence time (τ) sensitivity ────────────────────────────────────────
tau_cases = [30., 45., 60., 90., 120., 180.]
print('\n--- Residence time sensitivity  (T_in=200 °C, ini=0.005, cta=10) ---')
print(f'{"τ (s)":>7}  {"T_ss":>6}  {"X%":>6}  {"Mn":>8}  {"Mw":>8}  {"PDI":>6}  ok')
print('-' * 60)
tau_sens = {}
for tau_ in tau_cases:
    r = run_cstr(tau=tau_, rtol=1e-4, t_end_frac=6.)
    tau_sens[tau_] = r
    print(f'{tau_:7.0f}  {r["T_ss_C"]:6.1f}  {r["X_ss"]*100:6.2f}  '
          f'{r["Mn_ss"]:8.2f}  {r["Mw_ss"]:8.2f}  {r["PDI_ss"]:6.2f}  '
          f'{"ok" if r["ok"] else "FAIL"}')

# %%
# ── CTA sensitivity ───────────────────────────────────────────────────────
cta_cases = [0., 2., 5., 10., 20., 40., 80.]
print('\n--- CTA sensitivity  (τ=60 s, T_in=200 °C, ini=0.005) ---')
print(f'{"cta_in":>8}  {"T_ss":>6}  {"X%":>6}  {"Mn":>8}  {"Mw":>8}  {"PDI":>6}  ok')
print('-' * 64)
cta_sens = {}
for c_ in cta_cases:
    r = run_cstr(cta_in_=c_, rtol=1e-4, t_end_frac=6.)
    cta_sens[c_] = r
    print(f'{c_:8.1f}  {r["T_ss_C"]:6.1f}  {r["X_ss"]*100:6.2f}  '
          f'{r["Mn_ss"]:8.2f}  {r["Mw_ss"]:8.2f}  {r["PDI_ss"]:6.2f}  '
          f'{"ok" if r["ok"] else "FAIL"}')

# %%
# ── Initiator sensitivity ─────────────────────────────────────────────────
ini_cases = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050]
print('\n--- Initiator sensitivity  (τ=60 s, T_in=200 °C, cta=10) ---')
print(f'{"ini_in":>8}  {"T_ss":>6}  {"X%":>6}  {"Mn":>8}  {"Mw":>8}  {"PDI":>6}  ok')
print('-' * 64)
ini_sens = {}
for i_ in ini_cases:
    r = run_cstr(ini_in_=i_, rtol=1e-4, t_end_frac=6.)
    ini_sens[i_] = r
    print(f'{i_:8.4f}  {r["T_ss_C"]:6.1f}  {r["X_ss"]*100:6.2f}  '
          f'{r["Mn_ss"]:8.2f}  {r["Mw_ss"]:8.2f}  {r["PDI_ss"]:6.2f}  '
          f'{"ok" if r["ok"] else "FAIL"}')

# %%
# ── Feed temperature sensitivity ──────────────────────────────────────────
Tin_cases = [160., 170., 180., 190., 200., 210., 220., 230.]
print('\n--- Feed temperature sensitivity  (τ=60 s, ini=0.005, cta=10) ---')
print(f'{"T_in":>6}  {"T_ss":>6}  {"X%":>6}  {"Mn":>8}  {"Mw":>8}  {"PDI":>6}  ok')
print('-' * 62)
Tin_sens = {}
for Ti_ in Tin_cases:
    r = run_cstr(T_in_C=Ti_, rtol=1e-4, t_end_frac=6.)
    Tin_sens[Ti_] = r
    print(f'{Ti_:6.1f}  {r["T_ss_C"]:6.1f}  {r["X_ss"]*100:6.2f}  '
          f'{r["Mn_ss"]:8.2f}  {r["Mw_ss"]:8.2f}  {r["PDI_ss"]:6.2f}  '
          f'{"ok" if r["ok"] else "FAIL"}')

# %%
# ── Multiplicity: UA sweep (heat removal capability) ──────────────────────
# Scan UA to reveal possible multiple steady states (S-curve in T_ss vs UA)
UA_cases = [1000., 3000., 5000., 8000., 15000., 25000., 50000.]
print('\n--- Heat-removal sweep  (τ=60 s, T_in=200 °C, ini=0.005, cta=10) ---')
print(f'{"UA (W/K)":>10}  {"T_ss":>6}  {"X%":>6}  {"Mn":>8}  ok')
print('-' * 44)
UA_sens = {}
for UA_ in UA_cases:
    r = run_cstr(UA=UA_, rtol=1e-4, t_end_frac=8.)
    UA_sens[UA_] = r
    print(f'{UA_:10.0f}  {r["T_ss_C"]:6.1f}  {r["X_ss"]*100:6.2f}  '
          f'{r["Mn_ss"]:8.2f}  {"ok" if r["ok"] else "FAIL"}')

# %%
# ── Dynamic startup simulation ────────────────────────────────────────────
print('\n--- Dynamic startup (τ=60 s, base conditions, 600 s) ---')
base_dyn = run_cstr(t_end_frac=10., rtol=1e-5)

# Disturbance: step change in CTA at t = 3τ
print('\n--- Step disturbance: cta_in 10 → 30 mol/m³ at t = 3τ ---')

y_pre = base['y_ss'].copy()   # start from base SS
tau_d = 60.

def run_two_phase(tau, T_in_C, ini_in_, cta_before, cta_after,
                  t_switch, t_total, rtol=1e-5):
    """
    Run CSTR with a step change in cta_in at t_switch.
    Phase 1: [0, t_switch]  with cta_before
    Phase 2: [t_switch, t_total] with cta_after
    """
    T_in  = T_in_C + 273.15
    Tc_in = 130.   + 273.15
    atol_ = [1e-14,1e-11,1e-7,1e-11,1e-8,1e-3,1e-8,1e-1,1e-6,1e-4,1e-4]

    y0 = [0.,0.,0.,0.,0.,0., ini_in_, mono_in, cta_before, T_in, Tc_in]

    # Phase 1
    t1   = np.linspace(0., t_switch, 300)
    sol1 = solve_ivp(
        lambda t,y: cstr_odes(t,y,tau,T_in,Tc_in,ini_in_,mono_in,cta_before),
        (0.,t_switch), y0, method='Radau', t_eval=t1,
        rtol=rtol, atol=atol_, dense_output=False
    )
    # Phase 2 (start from end of Phase 1)
    t2   = np.linspace(t_switch, t_total, 300)
    sol2 = solve_ivp(
        lambda t,y: cstr_odes(t,y,tau,T_in,Tc_in,ini_in_,mono_in,cta_after),
        (t_switch,t_total), sol1.y[:,-1], method='Radau', t_eval=t2,
        rtol=rtol, atol=atol_, dense_output=False
    )
    t_all = np.concatenate([sol1.t, sol2.t[1:]])
    Y_all = np.hstack([sol1.y, sol2.y[:,1:]])
    return t_all, Y_all.T

t_disturb, Y_disturb = run_two_phase(
    tau=60., T_in_C=200., ini_in_=0.005,
    cta_before=10., cta_after=30.,
    t_switch=3*60., t_total=9*60.
)

# Compute profiles from Y_disturb
eps = 1e-30
m0d = Y_disturb[:,3]; m1d = Y_disturb[:,4]; m2d = Y_disturb[:,5]
Mn_d = Mw_mono*(m1d+eps)/(m0d+eps) / 1000.
Mw_d = Mw_mono*(m2d+eps)/(m1d+eps) / 1000.
X_d  = 1. - Y_disturb[:,7]/mono_in
T_d  = Y_disturb[:,9] - 273.15
Tc_d = Y_disturb[:,10]- 273.15
cta_d= Y_disturb[:,8]

print(f'  Before (t<3τ): Mn={Mn_d[299]:.1f} kg/mol  X={X_d[299]*100:.1f}%  T={T_d[299]:.1f}°C')
print(f'  After  (t>8τ): Mn={Mn_d[-1]:.1f} kg/mol  X={X_d[-1]*100:.1f}%  T={T_d[-1]:.1f}°C')

# %%
# ── QSSA check across temperature range ───────────────────────────────────
print('\n--- Rate constant summary at key temperatures ---')
print(f'{"T(°C)":>6}  {"kd":>10}  {"kp":>10}  {"Cs=ktr/kp":>10}  {"kβ":>10}')
print('-'*56)
for Tc_ in [180., 200., 220., 240., 260., 280.]:
    T_K  = Tc_ + 273.15
    kd_  = arrhenius(A_kd,      Ea_kd,      T_K)
    kp_  = arrhenius(A_kp,      Ea_kp,      T_K)
    kct_ = arrhenius(A_ktr_cta, Ea_ktr_cta, T_K)
    kbs_ = arrhenius(A_kbs,     Ea_kbs,     T_K)
    Cs_  = kct_ / kp_
    print(f'{Tc_:6.1f}  {kd_:10.4e}  {kp_:10.4e}  {Cs_:10.4f}  {kbs_:10.4e}')

# %%
# ── Property outputs ──────────────────────────────────────────────────────
def extract_profiles(sol_Y, mono_in_=mono_in):
    """Extract Mn, Mw, PDI, X, T, Tc from solution array (nt,11)."""
    eps = 1e-30
    m0 = sol_Y[:,3]; m1 = sol_Y[:,4]; m2 = sol_Y[:,5]
    Mn  = Mw_mono*(m1+eps)/(m0+eps) / 1000.
    Mw  = Mw_mono*(m2+eps)/(m1+eps) / 1000.
    PDI = Mw / np.maximum(Mn, eps)
    X   = 1. - sol_Y[:,7] / mono_in_
    T   = sol_Y[:,9]  - 273.15
    Tc  = sol_Y[:,10] - 273.15
    return Mn, Mw, PDI, X, T, Tc

# %%
# =============================================================================
# FIGURES
# =============================================================================

# ── Figure 1: Base-case dynamic startup ──────────────────────────────────
fig1, axes1 = plt.subplots(2, 4, figsize=(18, 8))
fig1.suptitle('LDPE CSTR step4 — Base-case startup dynamics\n'
              '(τ=60 s, T_in=200 °C, ini_in=0.005, cta_in=10 mol/m³, T_sp=240 °C)',
              fontsize=11)

t_b = base_dyn['t']
Y_b = base_dyn['Y']
Mn_b, Mw_b, PDI_b, X_b, T_b, Tc_b = extract_profiles(Y_b)

t_tau = t_b / tau_def   # time in units of τ

ax = axes1[0, 0]
ax.plot(t_tau, T_b, 'r', lw=2, label='T reactor')
ax.plot(t_tau, Tc_b,'b--', lw=1.5, label='Tc jacket')
ax.axhline(200., color='gray', ls=':', lw=0.8)
ax.axhline(280., color='gray', ls=':', lw=0.8)
ax.set(xlabel='t / τ', ylabel='T (°C)', title='Temperature dynamics')
ax.legend(fontsize=8)

ax = axes1[0, 1]
ax.plot(t_tau, X_b*100, 'g', lw=2)
ax.axhline(10., color='gray', ls=':', lw=0.8)
ax.axhline(30., color='gray', ls=':', lw=0.8)
ax.set(xlabel='t / τ', ylabel='Conversion (%)', title='Monomer conversion')

ax = axes1[0, 2]
ax.plot(t_tau, Mn_b, lw=2, label='Mn')
ax.plot(t_tau, Mw_b, lw=2, ls='--', label='Mw')
ax.axhline(20.,  color='gray', ls=':', lw=0.8)
ax.axhline(100., color='gray', ls=':', lw=0.8)
ax.set(xlabel='t / τ', ylabel='Molar mass (kg/mol)', title='Mn & Mw')
ax.legend(fontsize=8)

ax = axes1[0, 3]
ax.plot(t_tau, PDI_b, lw=2, color='purple')
ax.axhline(1.5, color='gray', ls=':', lw=0.8)
ax.axhline(3.0, color='gray', ls=':', lw=0.8)
ax.set(xlabel='t / τ', ylabel='PDI', title='Polydispersity index')

ax = axes1[1, 0]
ax.semilogy(t_tau, Y_b[:,0]+1e-20, lw=2, label='λ₀')
ax.semilogy(t_tau, Y_b[:,3]+1e-20, lw=2, ls='--', label='μ₀')
ax.set(xlabel='t / τ', ylabel='Moment (mol/m³)', title='λ₀ & μ₀ (radical / dead count)')
ax.legend(fontsize=8)

ax = axes1[1, 1]
ax.plot(t_tau, Y_b[:,6], lw=2, label='[I]', color='orange')
ax.plot(t_tau, Y_b[:,8], lw=2, ls='--', label='[CTA]', color='purple')
ax.set(xlabel='t / τ', ylabel='Concentration (mol/m³)', title='[I] and [CTA]')
ax.legend(fontsize=8)

ax = axes1[1, 2]
mono_norm = Y_b[:,7] / mono_in
ax.plot(t_tau, (1.-mono_norm)*100., 'g', lw=2)
ax.set(xlabel='t / τ', ylabel='Conversion (%)', title='Monomer conversion (detail)')

ax = axes1[1, 3]
# Phase portrait: T vs X
ax.plot(X_b*100, T_b, 'r', lw=2)
ax.plot(X_b[-1]*100, T_b[-1], 'ko', ms=8, label='SS')
ax.set(xlabel='Conversion (%)', ylabel='T (°C)', title='Phase portrait  T vs X')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('ldpe_cstr_v01_basecase.png', dpi=150)
print('\nFigure saved: ldpe_cstr_v01_basecase.png')

# ── Figure 2: Parametric studies ──────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8))
fig2.suptitle('LDPE CSTR step4 — Parametric sensitivity studies', fontsize=11)

# 2a: τ sensitivity — T_ss and X
tau_arr = np.array(tau_cases)
T_ss_tau = np.array([tau_sens[t]['T_ss_C']   for t in tau_cases])
X_ss_tau = np.array([tau_sens[t]['X_ss']      for t in tau_cases]) * 100.
Mn_ss_tau= np.array([tau_sens[t]['Mn_ss']     for t in tau_cases])
PDI_tau  = np.array([tau_sens[t]['PDI_ss']    for t in tau_cases])

ax = axes2[0, 0]
ax2b = ax.twinx()
ax.plot(tau_arr, T_ss_tau, 'ro-', lw=2, ms=6, label='T_ss (°C)')
ax2b.plot(tau_arr, X_ss_tau,'gs--',lw=2, ms=6, label='X (%)')
ax.axhline(200., color='r', ls=':', lw=0.7, alpha=0.5)
ax.set(xlabel='τ (s)', ylabel='T_ss (°C)', title='T & X vs τ')
ax2b.set_ylabel('X (%)', color='g')
ax2b.tick_params(axis='y', colors='g')
ax.legend(loc='upper left', fontsize=8)
ax2b.legend(loc='upper right', fontsize=8)

ax = axes2[0, 1]
ax.plot(tau_arr, Mn_ss_tau, 'bo-', lw=2, ms=6)
ax.axhline(20.,  color='gray', ls=':', lw=0.8)
ax.axhline(100., color='gray', ls=':', lw=0.8)
ax.set(xlabel='τ (s)', ylabel='Mn (kg/mol)', title='Mn vs τ')

# 2b: CTA sensitivity
cta_arr = np.array(cta_cases)
Mn_cta  = np.array([cta_sens[c]['Mn_ss']  for c in cta_cases])
Mw_cta  = np.array([cta_sens[c]['Mw_ss']  for c in cta_cases])
T_cta   = np.array([cta_sens[c]['T_ss_C'] for c in cta_cases])
X_cta   = np.array([cta_sens[c]['X_ss']   for c in cta_cases]) * 100.

ax = axes2[0, 2]
ax.plot(cta_arr, Mn_cta, 'bo-', lw=2, ms=6, label='Mn')
ax.plot(cta_arr, Mw_cta, 'rs--',lw=2, ms=6, label='Mw')
ax.axhline(20.,  color='gray', ls=':', lw=0.8)
ax.axhline(100., color='gray', ls=':', lw=0.8)
ax.set(xlabel='cta_in (mol/m³)', ylabel='Molar mass (kg/mol)',
       title='Mn & Mw vs CTA  (primary Mn lever)')
ax.legend(fontsize=8)

ax = axes2[0, 3]
ax2b = ax.twinx()
ax.plot(cta_arr, T_cta, 'ro-', lw=2, ms=6, label='T_ss')
ax2b.plot(cta_arr, X_cta,'gs--',lw=2, ms=6, label='X %')
ax.set(xlabel='cta_in (mol/m³)', ylabel='T_ss (°C)', title='T & X vs CTA')
ax2b.set_ylabel('X (%)', color='g')
ax2b.tick_params(axis='y', colors='g')
ax.legend(loc='center right', fontsize=8)
ax2b.legend(loc='upper right', fontsize=8)

# 2c: Initiator sensitivity
ini_arr = np.array(ini_cases)
Mn_ini  = np.array([ini_sens[i]['Mn_ss']  for i in ini_cases])
T_ini   = np.array([ini_sens[i]['T_ss_C'] for i in ini_cases])
X_ini   = np.array([ini_sens[i]['X_ss']   for i in ini_cases]) * 100.

ax = axes2[1, 0]
ax.semilogx(ini_arr, Mn_ini, 'bo-', lw=2, ms=6)
ax.axhline(20.,  color='gray', ls=':', lw=0.8)
ax.axhline(100., color='gray', ls=':', lw=0.8)
ax.set(xlabel='ini_in (mol/m³)', ylabel='Mn (kg/mol)', title='Mn vs ini_in')

ax = axes2[1, 1]
ax2b = ax.twinx()
ax.semilogx(ini_arr, T_ini, 'ro-', lw=2, ms=6, label='T_ss')
ax2b.semilogx(ini_arr, X_ini,'gs--',lw=2, ms=6, label='X %')
ax.set(xlabel='ini_in (mol/m³)', ylabel='T_ss (°C)', title='T & X vs ini_in')
ax2b.set_ylabel('X (%)', color='g')
ax2b.tick_params(axis='y', colors='g')
ax.legend(loc='upper left', fontsize=8)
ax2b.legend(loc='upper right', fontsize=8)

# 2d: Feed temperature sensitivity
Tin_arr = np.array(Tin_cases)
T_ss_Tin= np.array([Tin_sens[t]['T_ss_C']  for t in Tin_cases])
Mn_Tin  = np.array([Tin_sens[t]['Mn_ss']   for t in Tin_cases])
X_Tin   = np.array([Tin_sens[t]['X_ss']    for t in Tin_cases]) * 100.

ax = axes2[1, 2]
ax.plot(Tin_arr, T_ss_Tin - Tin_arr, 'ro-', lw=2, ms=6)
ax.axhline(0., color='k', lw=0.8, ls='--')
ax.set(xlabel='T_in (°C)', ylabel='T_ss − T_in (°C)',
       title='Adiabatic temperature rise vs T_in')

ax = axes2[1, 3]
ax2b = ax.twinx()
ax.plot(Tin_arr, Mn_Tin, 'bo-', lw=2, ms=6, label='Mn')
ax2b.plot(Tin_arr, X_Tin,'gs--',lw=2, ms=6, label='X %')
ax.axhline(20.,  color='gray', ls=':', lw=0.8)
ax.axhline(100., color='gray', ls=':', lw=0.8)
ax.set(xlabel='T_in (°C)', ylabel='Mn (kg/mol)', title='Mn & X vs T_in')
ax2b.set_ylabel('X (%)', color='g')
ax2b.tick_params(axis='y', colors='g')
ax.legend(loc='upper right', fontsize=8)
ax2b.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('ldpe_cstr_v01_parametric.png', dpi=150)
print('Figure saved: ldpe_cstr_v01_parametric.png')

# ── Figure 3: CTA step disturbance ────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))
fig3.suptitle('LDPE CSTR step4 — CTA step disturbance (cta_in: 10 → 30 mol/m³ at t=3τ,  T_in=200 °C, ini=0.005)',
              fontsize=11)

t_tau_d = t_disturb / 60.

ax = axes3[0]
ax.plot(t_tau_d, T_d,  'r', lw=2, label='T reactor')
ax.plot(t_tau_d, Tc_d, 'b--', lw=1.5, label='Tc jacket')
ax.axvline(3., color='orange', ls='--', lw=1.5, label='CTA step')
ax.set(xlabel='t / τ', ylabel='T (°C)', title='Reactor & jacket temperature')
ax.legend(fontsize=8)

ax = axes3[1]
ax.plot(t_tau_d, Mn_d, lw=2, label='Mn')
ax.plot(t_tau_d, Mw_d, lw=2, ls='--', label='Mw')
ax.axvline(3., color='orange', ls='--', lw=1.5)
ax.set(xlabel='t / τ', ylabel='Molar mass (kg/mol)', title='Mn & Mw response')
ax.legend(fontsize=8)

ax = axes3[2]
ax.plot(t_tau_d, X_d*100., 'g', lw=2)
ax.axvline(3., color='orange', ls='--', lw=1.5, label='CTA step')
ax.set(xlabel='t / τ', ylabel='Conversion (%)', title='Conversion response')
ax.legend(fontsize=8)

ax = axes3[3]
ax.plot(t_tau_d, cta_d, 'm', lw=2)
ax.axvline(3., color='orange', ls='--', lw=1.5, label='CTA step')
ax.set(xlabel='t / τ', ylabel='[CTA] (mol/m³)', title='CTA concentration profile')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('ldpe_cstr_v01_cta_disturbance.png', dpi=150)
print('Figure saved: ldpe_cstr_v01_cta_disturbance.png')

# ── Figure 4: Heat-removal multiplicity ───────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))
fig4.suptitle('LDPE CSTR step4 — Steady-state sensitivity to heat-removal (UA sweep)', fontsize=11)

UA_arr  = np.array(UA_cases)
T_UA    = np.array([UA_sens[u]['T_ss_C']  for u in UA_cases])
X_UA    = np.array([UA_sens[u]['X_ss']    for u in UA_cases]) * 100.
Mn_UA   = np.array([UA_sens[u]['Mn_ss']   for u in UA_cases])

ax = axes4[0]
ax.semilogx(UA_arr, T_UA, 'ro-', lw=2, ms=7)
ax.axhline(200., color='gray', ls=':', lw=0.8, label='200 °C')
ax.axhline(280., color='gray', ls=':', lw=0.8, label='280 °C')
ax.set(xlabel='UA (W/K)', ylabel='T_ss (°C)',
       title='Reactor T_ss vs heat-removal (UA)\n[potential for thermal runaway at low UA]')
ax.legend(fontsize=8)

ax = axes4[1]
ax.semilogx(UA_arr, X_UA, 'gs-', lw=2, ms=7)
ax.set(xlabel='UA (W/K)', ylabel='X (%)', title='Conversion vs UA')

ax = axes4[2]
ax.semilogx(UA_arr, Mn_UA, 'bo-', lw=2, ms=7)
ax.axhline(20.,  color='gray', ls=':', lw=0.8)
ax.axhline(100., color='gray', ls=':', lw=0.8)
ax.set(xlabel='UA (W/K)', ylabel='Mn (kg/mol)', title='Mn vs UA')

plt.tight_layout()
plt.savefig('ldpe_cstr_v01_ua_sensitivity.png', dpi=150)
print('Figure saved: ldpe_cstr_v01_ua_sensitivity.png')

# %%
# ── Final summary table ───────────────────────────────────────────────────
print('\n' + '=' * 65)
print('  CSTR Base-Case Summary')
print('=' * 65)
print(f'  Reactor volume V       = {V_cstr} m³')
print(f'  Residence time τ       = {tau_def} s')
print(f'  Heat transfer UA       = {UA_cstr:.0f} W/K')
print(f'  α_reactor              = {alpha_r:.5f} s⁻¹  (vs PFR h_r≈0.118 s⁻¹)')
print(f'  Feed: T_in={T_in_def-273.15:.0f}°C  [I]_in={ini_in:.4f}  [CTA]_in={cta_in}')
print()
print(f'  ── Steady-state results ──────────────────────────────')
print(f'  X_ss         = {base["X_ss"]*100:.2f} %       [target 10–30 %]')
print(f'  T_ss         = {base["T_ss_C"]:.1f} °C    [target 200–280 °C]')
print(f'  Tc_ss        = {base["Tc_ss_C"]:.1f} °C')
print(f'  Mn_ss        = {base["Mn_ss"]:.2f} kg/mol  [target 20–100]')
print(f'  Mw_ss        = {base["Mw_ss"]:.2f} kg/mol  [target 50–400]')
print(f'  PDI_ss       = {base["PDI_ss"]:.2f}          [target 1.5–3.0]')
print(f'  λ₀_ss        = {base["lam0_ss"]:.3e} mol/m³')
print('=' * 65)

plt.show()

# =============================================================================
# PHYSICAL BASIS NOTES
# =============================================================================
#
# CSTR vs PFR for LDPE:
#   The autoclave (CSTR) and tubular (PFR) reactors produce different LDPE grades:
#
#   Autoclave LDPE:
#     • Perfect mixing → broad MWD, broad short-chain branching distribution
#     • PDI typically 3–10 (can exceed PFR single-zone PDI ≈ 2)
#     • Higher short-chain branching (SCB) density → lower density LDPE
#     • Operating window: T = 180–280 °C, τ = 30–120 s
#     • Conversion per pass: 10–30%
#     • Used for film blowing, wire & cable, flexible packaging (high clarity)
#
#   Tubular LDPE (PFR, step2):
#     • Plug flow → narrower MWD at each axial position
#     • Axial T profile allows hot-spot control
#     • Higher LCB from ktrp (long chain branching) at high T zones
#     • Conversion per pass: 15–35% (better mass transfer driving force)
#
#   CSTR formulation note (β-scission):
#     In the PDF's CSTR balance, β-scission (kβ) is lumped into the combined
#     effective chain-transfer group K_eff = ktrm·[M] + ktr_CTA·[CTA] + kβ.
#     This treats β-scission as creating a new P₁ radical (chain termination +
#     re-initiation), which differs from the PFR model's symmetric random-
#     scission (k_bs·λ₁/2, k_bs·λ₂·2/3). The CSTR formulation is a simpler
#     but physically reasonable approximation for a well-mixed reactor.
#
#   Multiple steady states:
#     CSTR reactors can exhibit S-shaped temperature curves due to the
#     exothermic feedback: higher T → faster kp → more heat → higher T.
#     At low UA (poor cooling), the reactor can jump to a high-T runaway SS.
#     Control design must account for this non-linearity.
#
#   Moment closure for CSTR:
#     Same Hulburt–Katz closure as PFR is used for μ₃.
#     Note: in the CSTR the ktrp·(λ₁·μ₁ − λ₀·μ₂) term in dμ₁/dt can be
#     negative (dead chains are activated by LCB → reduce effective μ₁).
# =============================================================================
