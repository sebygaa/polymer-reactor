# %%
# Importing Packages
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# LDPE Free-Radical Polymerization — Method of Moments + Energy Balance
# Version 04: Dynamic cooling-water temperature (Tc as state variable)
#
# Changes from v03:
#   1. Tc is now the 10th state variable (y[9]) with its own ODE.
#      Previously Tc_const was a fixed parameter.
#   2. Cooling jacket energy balance added:
#        dTc/dt = (Tc_in − Tc)/τ_c  +  [4U/(D·ρc·Cp_c·β_j)] · (T − Tc)
#      where:
#        Tc_in  = coolant inlet temperature  [K]   (constant feed)
#        τ_c    = coolant residence time in jacket  [s]
#        ρc     = coolant density  [g/m³]
#        Cp_c   = coolant heat capacity  [J/(g·K)]
#        β_j    = jacket volume / reactor volume  [dimensionless]
#   3. Reactor energy balance unchanged in form; Tc_const replaced by y[9].
#
# Unit system:
#   Concentration : mol/m³
#   Rate constants: bimolecular → m³/(mol·s),  unimolecular (kd) → s⁻¹
#   Density       : g/m³
#   Heat capacity : J/(g·K)
#   Energy        : J/mol
# =============================================================================

# %%
# ---- Physical / Process Constants ----------------------------------------
# %%
R_gas   = 8.3145          # J/(mol·K)
Mw_mono = 28.054          # g/mol  (ethylene)
f_eff   = 0.8             # initiator efficiency

# Reactor geometry
D       = 0.05            # m  — tube inner diameter
U_heat  = 400.0           # W/(m²·K)

# Reaction mixture (supercritical ethylene/LDPE)
rho     = 600_000.0       # g/m³  (= 600 kg/m³)
Cp      = 1.7             # J/(g·K)
dH_p    = -93_000.0       # J/mol  (exothermic)

# %%
# ---- Cooling Jacket Parameters -------------------------------------------
# %%
# Pressurised water coolant at ~140 °C
Tc_in   = 140.0 + 273.15  # K  — coolant inlet temperature (constant)
tau_c   = 30.0            # s  — coolant residence time in jacket
                          #      (shorter → stronger control, e.g. 10–60 s)
rho_c   = 900_000.0       # g/m³  (= 900 kg/m³, pressurised hot water)
Cp_c    = 4.18            # J/(g·K)
beta_j  = 0.5             # Vj/Vr  — jacket-to-reactor volume ratio

# Pre-computed jacket heat-uptake coefficient [s⁻¹·K⁻¹ × K = s⁻¹]
# Term:  4U / (D · ρc · Cp_c · β_j)
alpha_j = 4.0 * U_heat / (D * rho_c * Cp_c * beta_j)   # s⁻¹

# %%
# ---- Arrhenius Parameters ------------------------------------------------
# %%
A_kd   = 3.15e15;  Ea_kd   = 155_000.0   # kd  [s⁻¹]
A_kp   = 6.58e4;   Ea_kp   =  29_500.0   # kp  [m³/(mol·s)]
A_ktc  = 2.0e5;    Ea_ktc  =   5_000.0   # ktc [m³/(mol·s)]
A_ktd  = 2.0e5;    Ea_ktd  =   5_000.0   # ktd [m³/(mol·s)]
A_ktrm = 1.5;      Ea_ktrm =  47_000.0   # ktrm[m³/(mol·s)]
A_ktrp = 3.0e-1;   Ea_ktrp =  50_000.0   # ktrp[m³/(mol·s)]


def arrhenius(A, Ea, T):
    return A * np.exp(-Ea / (R_gas * T))


# %%
# ---- ODE System -----------------------------------------------------------
# %%
# State vector y (10 elements):
#   y[0] = λ₀   live radical 0th moment  [mol/m³]
#   y[1] = λ₁   live radical 1st moment  [mol/m³]
#   y[2] = λ₂   live radical 2nd moment  [mol/m³]
#   y[3] = μ₀   dead polymer 0th moment  [mol/m³]
#   y[4] = μ₁   dead polymer 1st moment  [mol/m³]
#   y[5] = μ₂   dead polymer 2nd moment  [mol/m³]
#   y[6] = [I]  initiator concentration  [mol/m³]
#   y[7] = [M]  monomer concentration    [mol/m³]
#   y[8] = T    reactor temperature      [K]
#   y[9] = Tc   cooling water temperature[K]   ← NEW

def rxn_odes(t, y):
    lamb0, lamb1, lamb2 = y[0], y[1], y[2]
    mu0,   mu1,   mu2   = y[3], y[4], y[5]
    ini,   mono,  T, Tc = y[6], y[7], y[8], y[9]

    # --- Arrhenius rate constants ---
    kd   = arrhenius(A_kd,   Ea_kd,   T)
    kp   = arrhenius(A_kp,   Ea_kp,   T)
    ktc  = arrhenius(A_ktc,  Ea_ktc,  T)
    ktd  = arrhenius(A_ktd,  Ea_ktd,  T)
    ktrm = arrhenius(A_ktrm, Ea_ktrm, T)
    ktrp = arrhenius(A_ktrp, Ea_ktrp, T)

    # --- Hulburt & Katz closure for μ₃ ---
    eps = 1e-12
    if mu0 < eps or mu1 < eps:
        mu3 = 0.0
    else:
        mu3 = mu2 * (2.0 * mu0 * mu2 - mu1**2) / (mu0 * mu1)
    mu3 = max(mu3, 0.0)

    # --- Initiator and monomer ---
    dini_dt  = -kd * ini
    dmono_dt = -kp * mono * lamb0

    # --- Live radical moments ---
    dlamb0_dt = (2.0 * f_eff * kd * ini
                 - (ktc + ktd) * lamb0**2)

    dlamb1_dt = (kp   * mono * lamb0
                 + ktrm * mono * (lamb0 - lamb1)
                 + ktrp * (lamb0 * mu2 - lamb1 * mu1)
                 - (ktc + ktd) * lamb0 * lamb1)

    dlamb2_dt = (kp   * mono * (2.0 * lamb1 + lamb0)
                 + ktrm * mono * (lamb0 - lamb2)
                 + ktrp * (lamb0 * mu3 - lamb2 * mu1)
                 - (ktc + ktd) * lamb0 * lamb2)

    # --- Dead polymer moments ---
    dmu0_dt = (ktrm * mono * lamb0
               + (0.5 * ktc + ktd) * lamb0**2)

    dmu1_dt = (ktrm * mono * lamb1
               + ktrp * (lamb1 * mu1 - lamb0 * mu2)
               + (ktc + ktd) * lamb0 * lamb1)

    dmu2_dt = (ktrm * mono * lamb2
               + ktd  * lamb0 * lamb2
               + ktc  * (lamb0 * lamb2 + lamb1**2)
               + ktrp * (lamb2 * mu1 - lamb0 * mu3))

    # --- Reactor energy balance ---
    # dT/dt = [(-ΔHp)/(ρ·Cp)] · kp·[M]·λ₀  −  [4U/(ρ·Cp·D)] · (T − Tc)
    dT_dt = ((-dH_p) / (rho * Cp) * kp * mono * lamb0
             - 4.0 * U_heat / (rho * Cp * D) * (T - Tc))

    # --- Cooling jacket energy balance ---
    # dTc/dt = (Tc_in − Tc)/τ_c  +  α_j · (T − Tc)
    #   first term:  convective replacement of coolant
    #   second term: heat absorbed from reactor
    dTc_dt = ((Tc_in - Tc) / tau_c
              + alpha_j * (T - Tc))

    return [dlamb0_dt, dlamb1_dt, dlamb2_dt,
            dmu0_dt,   dmu1_dt,   dmu2_dt,
            dini_dt,   dmono_dt,  dT_dt, dTc_dt]


# %%
# ---- Initial Conditions ---------------------------------------------------
# %%
mono_0 = 2.005e4       # mol/m³   monomer
ini_0  = 100.0         # mol/m³   initiator
T_0    = 150 + 273.15  # K        reactor feed temperature
Tc_0   = Tc_in         # K        jacket starts at coolant inlet temperature

y0 = np.zeros(10)
y0[6] = ini_0
y0[7] = mono_0
y0[8] = T_0
y0[9] = Tc_0

# %%
# ---- Numerical Integration ------------------------------------------------
# %%
t_span = (0.0, 20.0)
t_eval = np.linspace(0.0, 20.0, 4000)

sol = solve_ivp(rxn_odes, t_span, y0,
                method='Radau',
                t_eval=t_eval,
                rtol=1e-6, atol=1e-10,
                dense_output=False)

t_res = sol.t
y_res = sol.y.T   # shape: (n_points, 10)

# %%
# ---- Post-processing ------------------------------------------------------
# %%
lamb0_res = y_res[:, 0]
lamb1_res = y_res[:, 1]
lamb2_res = y_res[:, 2]
mu0_res   = y_res[:, 3]
mu1_res   = y_res[:, 4]
mu2_res   = y_res[:, 5]
ini_res   = y_res[:, 6]
mono_res  = y_res[:, 7]
T_res     = y_res[:, 8]
Tc_res    = y_res[:, 9]

eps = 1e-30
lam0_total = lamb0_res + mu0_res + eps
lam1_total = lamb1_res + mu1_res + eps
lam2_total = lamb2_res + mu2_res + eps

Mn_res     = Mw_mono * lam1_total / lam0_total
Mw_res     = Mw_mono * lam2_total / lam1_total
PDI_res    = Mw_res / Mn_res
conversion = 1.0 - mono_res / mono_0

# %%
# ---- Plots ----------------------------------------------------------------
# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle('LDPE Free-Radical Polymerization — MoM Simulation (v04, dynamic Tc)', fontsize=11)

axes[0, 0].plot(t_res, mono_res / mono_0 * 100)
axes[0, 0].set_ylabel('Monomer remaining (%)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_title('Monomer conversion')

axes[0, 1].plot(t_res, T_res  - 273.15, color='tomato', label='Reactor T')
axes[0, 1].plot(t_res, Tc_res - 273.15, color='steelblue', label='Coolant Tc')
axes[0, 1].axhline(Tc_in - 273.15, color='steelblue', linestyle='--', alpha=0.5, label='Tc_in')
axes[0, 1].set_ylabel('Temperature (°C)')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_title('Reactor & coolant temperature')
axes[0, 1].legend(fontsize=8)

axes[0, 2].plot(t_res, T_res - Tc_res, color='darkorange')
axes[0, 2].set_ylabel('T − Tc  (K)')
axes[0, 2].set_xlabel('Time (s)')
axes[0, 2].set_title('Temperature driving force')

axes[0, 3].plot(t_res, ini_res)
axes[0, 3].set_ylabel('[I] (mol/m³)')
axes[0, 3].set_xlabel('Time (s)')
axes[0, 3].set_title('Initiator concentration')

axes[1, 0].plot(t_res, Mn_res / 1000)
axes[1, 0].set_ylabel('Mn (kg/mol)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_title('Number-avg mol. weight')

axes[1, 1].plot(t_res, Mw_res / 1000)
axes[1, 1].set_ylabel('Mw (kg/mol)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_title('Weight-avg mol. weight')

axes[1, 2].plot(t_res, PDI_res)
axes[1, 2].set_ylabel('PDI (—)')
axes[1, 2].set_xlabel('Time (s)')
axes[1, 2].set_title('Polydispersity index')

axes[1, 3].plot(t_res, conversion * 100)
axes[1, 3].set_ylabel('Conversion (%)')
axes[1, 3].set_xlabel('Time (s)')
axes[1, 3].set_title('Monomer conversion')

plt.tight_layout()
plt.savefig('ldpe_simulation_v04.png', dpi=150)
plt.show()

# %%
# ---- Print summary --------------------------------------------------------
# %%
print('=== Simulation Summary (t = {:.2f} s) ==='.format(t_res[-1]))
print(f'  Monomer conversion : {conversion[-1]*100:.1f} %')
print(f'  Reactor temp  T    : {T_res[-1]-273.15:.1f} °C')
print(f'  Coolant temp  Tc   : {Tc_res[-1]-273.15:.1f} °C')
print(f'  T − Tc (final)     : {T_res[-1]-Tc_res[-1]:.1f} K')
print(f'  Mn                 : {Mn_res[-1]/1000:.2f} kg/mol')
print(f'  Mw                 : {Mw_res[-1]/1000:.2f} kg/mol')
print(f'  PDI                : {PDI_res[-1]:.2f}')
