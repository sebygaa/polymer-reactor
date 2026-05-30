# %%
# Importing Packages
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# LDPE Free-Radical Polymerization — Method of Moments + Energy Balance
# Version 03: Arrhenius temperature-dependent rate constants
#
# Changes from v02:
#   1. All six rate constants replaced by Arrhenius functions k(T) = A·exp(-Ea/RT).
#      Parameters sourced from Asteasuain et al. (2001), Buback et al. (1994),
#      Tobita (1993).  See inline references.
#   2. Fixed initial-condition swap: [I]→y[6], [M]→y[7]  (v02 had these reversed).
#   3. Fixed μ₃ closure guard: epsilon-additive scheme replaced by conditional
#      check as recommended in model doc v4.
#   4. rho set to literature-consistent constant (ideal-gas calculation removed;
#      ideal gas over-predicts density ~4× at supercritical LDPE conditions).
#   5. Switched integrator from odeint to solve_ivp(Radau) for stiff ODEs.
#
# Thermal note:
#   With these parameters the reactor may exhibit thermal runaway (adiabatic
#   ΔT for 30 % conversion ≈ 480 K).  This is physically consistent with
#   uncontrolled LDPE reactors.  To represent a controlled process, reduce D
#   (higher wall-area/volume ratio), increase U, or tune the initiator Ea/A.
#
# Unit system (must be consistent throughout):
#   Concentration : mol/m³
#   Rate constants: bimolecular → m³/(mol·s),  unimolecular (kd) → s⁻¹
#   Density (rho) : g/m³
#   Heat capacity : J/(g·K)
#   Energy        : J/mol
# =============================================================================

# %%
# ---- Physical / Process Constants ----------------------------------------
# %%
R_gas   = 8.3145          # J/(mol·K)
Mw_mono = 28.054          # g/mol  (ethylene, CH2=CH2)
f_eff   = 0.8             # initiator efficiency (dimensionless, 0.5–0.8)

# Reactor geometry
D       = 0.05            # m  — tube inner diameter (typical: 0.02–0.05 m)
U_heat  = 400.0           # W/(m²·K) = J/(s·m²·K)  — overall HTC

# Thermal properties of reaction mixture
# rho: supercritical ethylene at ~200 °C, ~2000–3000 bar
#   Ideal gas gives ~2400 kg/m³ (grossly overpredicts at these conditions).
#   PVT data / NIST webbook: ~500–700 kg/m³  → use 600 kg/m³ as representative.
rho     = 600_000.0       # g/m³  (= 600 kg/m³)

# Cp: ethylene monomer heat capacity at high pressure
#   Gas-phase Cp(C2H4) ≈ 42.9 J/(mol·K) = 1.53 J/(g·K)
#   Polymer (LDPE) Cp ≈ 2.1 J/(g·K)
#   Mixture (early conversion) ≈ 1.6–2.0 J/(g·K) → use 1.7 J/(g·K)
Cp      = 1.7             # J/(g·K)

# Heat of polymerization of ethylene
# Exothermic: ΔH_p ≈ −93 to −96 kJ/mol  (sign: negative = exothermic)
dH_p    = -93_000.0       # J/mol

# Cooling jacket temperature (constant — simplifying assumption)
# Must be lower than T_0 for the jacket to act as a coolant.
# Typical LDPE: feed enters at 150 °C, jacket at ~130–150 °C.
Tc_const = 140 + 273.15   # K  (140 °C)

# %%
# ---- Arrhenius Parameters ------------------------------------------------
# %%
# Reference:
#   kd  — Asteasuain et al. (2001), Brown et al. (2002)
#           representative DTBP-type high-temperature peroxide
#   kp  — Buback et al. (1994) Macromol. Chem. Phys. 196, 3267
#           high-pressure ethylene (pressure correction excluded)
#   ktc, ktd — Asteasuain et al. (2001), simplified (no gel-effect model)
#   ktrm — Bamford & Tompa (1953), adapted for ethylene
#   ktrp — Tobita (1993), Asteasuain et al. (2001)
#
# All bimolecular A values in m³/(mol·s).
# Temperature range validity: ~150–300 °C (423–573 K), 1000–3000 bar.

# Initiator decomposition  [s⁻¹]
A_kd   = 3.15e15          # s⁻¹
Ea_kd  = 155_000.0        # J/mol   (155 kJ/mol)

# Propagation  [m³/(mol·s)]
A_kp   = 6.58e4           # m³/(mol·s)
Ea_kp  = 29_500.0         # J/mol   ( 29.5 kJ/mol, Buback 1994)

# Termination by combination  [m³/(mol·s)]
A_ktc  = 2.0e5            # m³/(mol·s)
Ea_ktc = 5_000.0          # J/mol

# Termination by disproportionation  [m³/(mol·s)]
A_ktd  = 2.0e5            # m³/(mol·s)
Ea_ktd = 5_000.0          # J/mol

# Chain transfer to monomer  [m³/(mol·s)]
A_ktrm  = 1.5             # m³/(mol·s)
Ea_ktrm = 47_000.0        # J/mol   ( 47 kJ/mol)

# Chain transfer to polymer (LCB)  [m³/(mol·s)]
A_ktrp  = 3.0e-1          # m³/(mol·s)
Ea_ktrp = 50_000.0        # J/mol   ( 50 kJ/mol)


def arrhenius(A, Ea, T):
    """Return rate constant at temperature T [K]."""
    return A * np.exp(-Ea / (R_gas * T))


# %%
# ---- ODE System -----------------------------------------------------------
# %%
# State vector y (9 elements):
#   y[0] = λ₀   live radical 0th moment  [mol/m³]
#   y[1] = λ₁   live radical 1st moment  [mol/m³]
#   y[2] = λ₂   live radical 2nd moment  [mol/m³]
#   y[3] = μ₀   dead polymer 0th moment  [mol/m³]
#   y[4] = μ₁   dead polymer 1st moment  [mol/m³]
#   y[5] = μ₂   dead polymer 2nd moment  [mol/m³]
#   y[6] = [I]  initiator concentration  [mol/m³]
#   y[7] = [M]  monomer concentration    [mol/m³]
#   y[8] = T    reactor temperature      [K]

def rxn_odes(t, y):
    lamb0, lamb1, lamb2 = y[0], y[1], y[2]
    mu0,   mu1,   mu2   = y[3], y[4], y[5]
    ini,   mono,  T     = y[6], y[7], y[8]

    # --- Arrhenius: update all rate constants at current T ---
    kd   = arrhenius(A_kd,   Ea_kd,   T)
    kp   = arrhenius(A_kp,   Ea_kp,   T)
    ktc  = arrhenius(A_ktc,  Ea_ktc,  T)
    ktd  = arrhenius(A_ktd,  Ea_ktd,  T)
    ktrm = arrhenius(A_ktrm, Ea_ktrm, T)
    ktrp = arrhenius(A_ktrp, Ea_ktrp, T)

    # --- Hulburt & Katz closure: μ₃ ≈ μ₂(2μ₀μ₂ − μ₁²)/(μ₀·μ₁) ---
    eps = 1e-12
    if mu0 < eps or mu1 < eps:
        mu3 = 0.0
    else:
        mu3 = mu2 * (2.0 * mu0 * mu2 - mu1**2) / (mu0 * mu1)
    mu3 = max(mu3, 0.0)   # prevent negative values from numerical noise

    # --- Initiator and monomer ---
    dini_dt  = -kd * ini
    dmono_dt = -kp * mono * lamb0

    # --- Live radical moments ---
    # λ₀
    dlamb0_dt = (2.0 * f_eff * kd * ini
                 - (ktc + ktd) * lamb0**2)

    # λ₁
    dlamb1_dt = (kp   * mono * lamb0
                 + ktrm * mono * (lamb0 - lamb1)
                 + ktrp * (lamb0 * mu2 - lamb1 * mu1)
                 - (ktc + ktd) * lamb0 * lamb1)

    # λ₂
    dlamb2_dt = (kp   * mono * (2.0 * lamb1 + lamb0)
                 + ktrm * mono * (lamb0 - lamb2)
                 + ktrp * (lamb0 * mu3 - lamb2 * mu1)
                 - (ktc + ktd) * lamb0 * lamb2)

    # --- Dead polymer moments ---
    # μ₀  (LCB has zero net effect on chain count → no ktrp term)
    dmu0_dt = (ktrm * mono * lamb0
               + (0.5 * ktc + ktd) * lamb0**2)

    # μ₁
    dmu1_dt = (ktrm * mono * lamb1
               + ktrp * (lamb1 * mu1 - lamb0 * mu2)
               + (ktc + ktd) * lamb0 * lamb1)

    # μ₂
    dmu2_dt = (ktrm * mono * lamb2
               + ktd  * lamb0 * lamb2
               + ktc  * (lamb0 * lamb2 + lamb1**2)
               + ktrp * (lamb2 * mu1 - lamb0 * mu3))

    # --- Energy balance ---
    # dT/dτ = [(-ΔHp)/(ρ·Cp)] · kp·[M]·λ₀  −  [4U/(ρ·Cp·D)] · (T − Tc)
    # Units check (all m³-based):
    #   generation: [J/mol]/([g/m³]·[J/(g·K)]) · [m³/(mol·s)]·[mol/m³]·[mol/m³] = K/s ✓
    #   cooling:    [W/(m²·K)] / ([g/m³]·[J/(g·K)]·[m])                          = 1/s ✓
    dT_dt = ((-dH_p) / (rho * Cp) * kp * mono * lamb0
             - 4.0 * U_heat / (rho * Cp * D) * (T - Tc_const))

    return [dlamb0_dt, dlamb1_dt, dlamb2_dt,
            dmu0_dt,   dmu1_dt,   dmu2_dt,
            dini_dt,   dmono_dt,  dT_dt]


# %%
# ---- Initial Conditions ---------------------------------------------------
# %%
# Operating pressure ~2000 bar, T₀ = 150 °C
# [M]₀: supercritical ethylene density ~600 kg/m³ → 600,000 g/m³ / 28.054 g/mol ≈ 21,390 mol/m³
mono_0 = 2.005e4    # mol/m³   monomer (ethylene)
ini_0  = 100.0      # mol/m³   initiator (~0.5 % of monomer)
T_0    = 150 + 273.15   # K

y0 = np.zeros(9)
y0[0] = 0.0       # λ₀
y0[1] = 0.0       # λ₁
y0[2] = 0.0       # λ₂
y0[3] = 0.0       # μ₀
y0[4] = 0.0       # μ₁
y0[5] = 0.0       # μ₂
y0[6] = ini_0     # [I]   ← fixed (was swapped with mono in v02)
y0[7] = mono_0    # [M]   ← fixed
y0[8] = T_0       # T

# %%
# ---- Numerical Integration ------------------------------------------------
# %%
# Use Radau (implicit, A-stable) for stiff ODEs as recommended in model doc v4.
t_span = (0.0, 20.0)
t_eval = np.linspace(0.0, 20.0, 4000)

sol = solve_ivp(rxn_odes, t_span, y0,
                method='Radau',
                t_eval=t_eval,
                rtol=1e-6, atol=1e-10,
                dense_output=False)

t_res = sol.t
y_res = sol.y.T     # shape: (n_points, 9)

# %%
# ---- Post-processing: Molecular Weights -----------------------------------
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

# Combine live + dead moments for overall MWD
eps = 1e-30
lam0_total = lamb0_res + mu0_res + eps
lam1_total = lamb1_res + mu1_res + eps
lam2_total = lamb2_res + mu2_res + eps

Mn_res  = Mw_mono * lam1_total / lam0_total           # g/mol
Mw_res  = Mw_mono * lam2_total / lam1_total           # g/mol
PDI_res = Mw_res / Mn_res

conversion = 1.0 - mono_res / mono_0

# %%
# ---- Plots ----------------------------------------------------------------
# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle('LDPE Free-Radical Polymerization — MoM Simulation (v03)', fontsize=11)

axes[0, 0].plot(t_res, mono_res / mono_0 * 100)
axes[0, 0].set_ylabel('Monomer remaining (%)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_title('Monomer conversion')

axes[0, 1].plot(t_res, T_res - 273.15, color='tomato')
axes[0, 1].set_ylabel('Temperature (°C)')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_title('Reactor temperature')

axes[0, 2].plot(t_res, ini_res)
axes[0, 2].set_ylabel('[I] (mol/m³)')
axes[0, 2].set_xlabel('Time (s)')
axes[0, 2].set_title('Initiator concentration')

axes[1, 0].plot(t_res, Mn_res / 1000)
axes[1, 0].set_ylabel('Mn (kg/mol)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_title('Number-avg molecular weight')

axes[1, 1].plot(t_res, Mw_res / 1000)
axes[1, 1].set_ylabel('Mw (kg/mol)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_title('Weight-avg molecular weight')

axes[1, 2].plot(t_res, PDI_res)
axes[1, 2].set_ylabel('PDI (—)')
axes[1, 2].set_xlabel('Time (s)')
axes[1, 2].set_title('Polydispersity index')

plt.tight_layout()
plt.savefig('ldpe_simulation_v03.png', dpi=150)
plt.show()

# %%
# ---- Print summary at end of simulation ----------------------------------
# %%
print('=== Simulation Summary (t = {:.2f} s) ==='.format(t_res[-1]))
print(f'  Monomer conversion : {conversion[-1]*100:.1f} %')
print(f'  Temperature        : {T_res[-1]-273.15:.1f} °C')
print(f'  Mn                 : {Mn_res[-1]/1000:.2f} kg/mol')
print(f'  Mw                 : {Mw_res[-1]/1000:.2f} kg/mol')
print(f'  PDI                : {PDI_res[-1]:.2f}')
