# %%
# Importing Packages
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# LDPE Free-Radical Polymerization — Dynamic 1D PFR Model
# Version 01: Method of Lines (MOL) + Upwind Finite Difference (Upwind FDM)
#
# Governing PDEs (per species C):
#   ∂C/∂t  + v  · ∂C/∂z  = R(C, T)        [reactor fluid, flow in +z]
#   ∂Tc/∂t − vc · ∂Tc/∂z = h_j · (T − Tc) [jacket coolant, counter-current, flow in −z]
#
# Spatial discretization — Upwind scheme (stability for hyperbolic PDEs):
#   Reactor  (+z flow, v > 0)  → Backward difference: ∂C/∂z  ≈ (C[i] − C[i−1]) / dz
#   Jacket   (−z flow, vc > 0) → Forward  difference: ∂Tc/∂z ≈ (Tc[i+1] − Tc[i]) / dz
#
# Boundary conditions:
#   z = 0  (i = 0):   reactor inlet  [I]=ini_0, [M]=mono_0, T=T_0, moments=0  (fixed)
#   z = L  (i = N−1): coolant inlet  Tc = Tc_in                                (fixed)
#
# State vector layout: y[i*10 + j]   (i = spatial node, j = variable index)
#   j=0 λ₀, j=1 λ₁, j=2 λ₂   — live radical moments   [mol/m³]
#   j=3 μ₀, j=4 μ₁, j=5 μ₂   — dead polymer moments   [mol/m³]
#   j=6 [I], j=7 [M]           — concentrations         [mol/m³]
#   j=8 T,   j=9 Tc             — temperatures           [K]
#
# Unit system:
#   Concentration : mol/m³
#   Rate constants: bimolecular → m³/(mol·s),  unimolecular (kd) → s⁻¹
#   Density       : g/m³
#   Heat capacity : J/(g·K)
#   Energy        : J/mol
# =============================================================================

# %%
# ---- Physical Constants and Parameters ------------------------------------
# %%
R_gas   = 8.3145          # J/(mol·K)
Mw_mono = 28.054          # g/mol  (ethylene)
f_eff   = 0.8             # initiator efficiency

# Reactor geometry
L     = 500.0             # m   — tube length
D     = 0.05              # m   — reactor inner diameter
D_j   = 0.07              # m   — jacket inner diameter (concentric tube)
A_r   = np.pi / 4 * D**2                    # m²  reactor cross-section
A_c   = np.pi / 4 * (D_j**2 - D**2)        # m²  jacket cross-section

# Flow velocities
v     = 12.0              # m/s — reactor fluid (superficial)
v_c   = 1.0               # m/s — coolant velocity magnitude (direction: −z)

# Thermal properties of reaction mixture
rho   = 600_000.0         # g/m³  (= 600 kg/m³, supercritical ethylene/LDPE)
Cp    = 1.7               # J/(g·K)
dH_p  = -93_000.0         # J/mol  (exothermic)

# Overall heat-transfer coefficient
U_heat = 400.0            # W/(m²·K)

# Jacket coolant (pressurised hot water)
Tc_in  = 140.0 + 273.15   # K  — coolant inlet temperature (at z = L)
rho_c  = 900_000.0         # g/m³  (= 900 kg/m³)
Cp_c   = 4.18              # J/(g·K)

# Heat-transfer rate coefficients [s⁻¹]:
#   h_r = 4U / (ρ·Cp·D)             — reactor-side cooling   [K/s per K of T−Tc]
#   h_j = U·π·D / (ρc·Cp_c·Ac)     — jacket-side heating    [K/s per K of T−Tc]
h_r = 4.0 * U_heat / (rho * Cp * D)
h_j = U_heat * np.pi * D / (rho_c * Cp_c * A_c)

# %%
# ---- Arrhenius Parameters -------------------------------------------------
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
# ---- Spatial Grid ---------------------------------------------------------
# %%
N  = 40                         # number of spatial nodes
z  = np.linspace(0.0, L, N)    # node positions [m]
dz = z[1] - z[0]               # grid spacing   [m]
NV = 10                         # state variables per node

print(f'Grid: N={N}, dz={dz:.1f} m, L={L:.0f} m')
print(f'Nominal residence time τ = {L/v:.1f} s')

# %%
# ---- Inlet / Boundary Conditions ------------------------------------------
# %%
mono_0 = 2.005e4        # mol/m³  monomer at inlet
ini_0  = 100.0          # mol/m³  initiator at inlet
T_0    = 150.0 + 273.15 # K       reactor inlet temperature
# Coolant BC: Tc(z=L, t) = Tc_in  (fixed)
# Reactor BC: all variables at i=0 are fixed to inlet values

# %%
# ---- ODE Function (Method of Lines) ---------------------------------------
# %%
def pfr_odes(t, y):
    """
    Method-of-Lines RHS for the 1-D dynamic PFR.

    Spatial upwind FDM:
      - Reactor variables (j=0..8): backward difference (flow +z)
      - Jacket Tc        (j=9):     forward  difference (flow −z)
    """
    s = y.reshape(N, NV)

    lamb0 = s[:, 0];  lamb1 = s[:, 1];  lamb2 = s[:, 2]
    mu0   = s[:, 3];  mu1   = s[:, 4];  mu2   = s[:, 5]
    ini   = s[:, 6];  mono  = s[:, 7]
    T     = s[:, 8];  Tc    = s[:, 9]

    # --- Arrhenius rate constants (vectorised, shape N) ---
    kd   = arrhenius(A_kd,   Ea_kd,   T)
    kp   = arrhenius(A_kp,   Ea_kp,   T)
    ktc  = arrhenius(A_ktc,  Ea_ktc,  T)
    ktd  = arrhenius(A_ktd,  Ea_ktd,  T)
    ktrm = arrhenius(A_ktrm, Ea_ktrm, T)
    ktrp = arrhenius(A_ktrp, Ea_ktrp, T)

    # --- Hulburt & Katz μ₃ closure ---
    eps3 = 1e-12
    mu3 = np.where(
        (mu0 < eps3) | (mu1 < eps3),
        0.0,
        np.maximum(mu2 * (2.0 * mu0 * mu2 - mu1**2) / (mu0 * mu1 + eps3), 0.0)
    )

    # --- Reaction source terms R [shape N] ---
    R_l0 = (2.0 * f_eff * kd * ini
            - (ktc + ktd) * lamb0**2)

    R_l1 = (kp * mono * lamb0
            + ktrm * mono * (lamb0 - lamb1)
            + ktrp * (lamb0 * mu2 - lamb1 * mu1)
            - (ktc + ktd) * lamb0 * lamb1)

    R_l2 = (kp * mono * (2.0 * lamb1 + lamb0)
            + ktrm * mono * (lamb0 - lamb2)
            + ktrp * (lamb0 * mu3 - lamb2 * mu1)
            - (ktc + ktd) * lamb0 * lamb2)

    R_m0 = (ktrm * mono * lamb0
            + (0.5 * ktc + ktd) * lamb0**2)

    R_m1 = (ktrm * mono * lamb1
            + ktrp * (lamb1 * mu1 - lamb0 * mu2)
            + (ktc + ktd) * lamb0 * lamb1)

    R_m2 = (ktrm * mono * lamb2
            + ktd * lamb0 * lamb2
            + ktc * (lamb0 * lamb2 + lamb1**2)
            + ktrp * (lamb2 * mu1 - lamb0 * mu3))

    R_ini  = -kd * ini
    R_mono = -kp * mono * lamb0

    R_T    = ((-dH_p) / (rho * Cp) * kp * mono * lamb0
              - h_r * (T - Tc))

    R_Tc   = h_j * (T - Tc)

    # --- Build dydt array ---
    dydt = np.zeros_like(s)

    # -------------------------------------------------------------------
    # Reactor variables j=0..8: flow in +z  → Backward upwind FDM
    #   dC/dz ≈ (C[i] − C[i−1]) / dz
    #   i=0: fixed inlet BC  →  dydt[0, j] = 0
    #   i=1..N−1: upwind difference with upstream ghost = inlet BC at i=0
    # -------------------------------------------------------------------
    reactor_vars  = [lamb0, lamb1, lamb2, mu0,   mu1,  mu2,  ini,   mono,  T  ]
    reactor_rhs   = [R_l0,  R_l1,  R_l2,  R_m0, R_m1, R_m2, R_ini, R_mono, R_T]
    reactor_bcs   = [0.0,   0.0,   0.0,   0.0,  0.0,  0.0,  ini_0, mono_0, T_0]

    for j, (C, R, BC) in enumerate(zip(reactor_vars, reactor_rhs, reactor_bcs)):
        # Build upstream array: C_up[i] = C[i−1], with C_up[0] = BC
        C_up = np.empty(N)
        C_up[0]  = BC       # inlet ghost cell
        C_up[1:] = C[:-1]   # interior: shift by 1

        # i=0: inlet BC fixed
        dydt[0, j] = 0.0
        # i=1..N−1: backward upwind
        dCdz = (C - C_up) / dz
        dydt[1:, j] = R[1:] - v * dCdz[1:]

    # -------------------------------------------------------------------
    # Jacket coolant j=9: flow in −z  → Forward upwind FDM
    #   ∂Tc/∂t − vc·∂Tc/∂z = R_Tc
    #   Upwind for −z advection: ∂Tc/∂z ≈ (Tc[i+1] − Tc[i]) / dz
    #   → dTc/dt = R_Tc + vc·(Tc[i+1] − Tc[i]) / dz
    #   i=N−1: fixed coolant inlet BC  →  dydt[N−1, 9] = 0
    #   i=0..N−2: forward upwind with downstream ghost = Tc_in at i=N−1
    # -------------------------------------------------------------------
    Tc_dn = np.empty(N)
    Tc_dn[:-1] = Tc[1:]    # interior: shift by 1
    Tc_dn[-1]  = Tc_in     # outlet ghost cell (coolant inlet BC)

    dTcdz = (Tc_dn - Tc) / dz
    dydt[:N-1, 9] = R_Tc[:N-1] + v_c * dTcdz[:N-1]
    dydt[N-1,  9] = 0.0    # coolant inlet: fixed BC

    return dydt.ravel()


# %%
# ---- Initial Conditions ---------------------------------------------------
# %%
# At t=0: reactor is filled with monomer at feed temperature.
# No initiator in reactor body (only arrives at z=0 via feed).
# Coolant jacket pre-filled at Tc_in uniformly.
y0 = np.zeros((N, NV))
y0[:, 6] = 0.0      # [I] = 0 initially (no initiator in reactor at t=0)
y0[:, 7] = mono_0   # [M] = mono_0 (reactor filled with monomer)
y0[:, 8] = T_0      # T   = T_0
y0[:, 9] = Tc_in    # Tc  = Tc_in (jacket pre-filled at coolant inlet temp)

# Impose BCs at t=0
y0[0, 6]    = ini_0   # reactor inlet [I]
y0[N-1, 9]  = Tc_in   # coolant inlet Tc

# %%
# ---- Time Integration -----------------------------------------------------
# %%
tau_res = L / v              # residence time [s]
t_end   = 4.0 * tau_res      # simulate 4 residence times to approach steady state
t_eval  = np.linspace(0.0, t_end, 300)

print(f'\nSimulating 0 → {t_end:.0f} s  ({t_end/tau_res:.1f} × τ) ...')

sol = solve_ivp(
    pfr_odes,
    (0.0, t_end),
    y0.ravel(),
    method='Radau',
    t_eval=t_eval,
    rtol=1e-5,
    atol=1e-9,
    dense_output=False,
)

print(f'Done. status={sol.status}, nfev={sol.nfev}')
print(f'Message: {sol.message}')

# %%
# ---- Post-processing -------------------------------------------------------
# %%
nt = len(sol.t)
Y  = sol.y.T.reshape(nt, N, NV)   # (nt, N, 10)

lamb0_s = Y[:, :, 0];  lamb1_s = Y[:, :, 1];  lamb2_s = Y[:, :, 2]
mu0_s   = Y[:, :, 3];  mu1_s   = Y[:, :, 4];  mu2_s   = Y[:, :, 5]
ini_s   = Y[:, :, 6];  mono_s  = Y[:, :, 7]
T_s     = Y[:, :, 8];  Tc_s    = Y[:, :, 9]

eps_mw  = 1e-30
l0_tot  = lamb0_s + mu0_s + eps_mw
l1_tot  = lamb1_s + mu1_s + eps_mw
l2_tot  = lamb2_s + mu2_s + eps_mw

Mn_s    = Mw_mono * l1_tot / l0_tot    # g/mol
Mw_s    = Mw_mono * l2_tot / l1_tot    # g/mol
PDI_s   = Mw_s / Mn_s
X_s     = 1.0 - mono_s / mono_0        # conversion

# %%
# ---- Figure 1: Steady-state spatial profiles (final time) -----------------
# %%
fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8))
fig1.suptitle(
    f'LDPE PFR — Steady-state profiles  (t = {sol.t[-1]:.0f} s ≈ {sol.t[-1]/tau_res:.1f} τ)',
    fontsize=11
)

# Temperature
ax = axes1[0, 0]
ax.plot(z, T_s[-1] - 273.15,  'r',  lw=2, label='Reactor T')
ax.plot(z, Tc_s[-1] - 273.15, 'b--', lw=2, label='Coolant Tc')
ax.axhline(Tc_in - 273.15, color='b', lw=0.8, alpha=0.5, linestyle=':',
           label=f'Tc_in={Tc_in-273.15:.0f} °C')
ax.set_xlabel('z (m)');  ax.set_ylabel('T (°C)')
ax.set_title('Temperature profiles');  ax.legend(fontsize=8)

# Conversion
axes1[0, 1].plot(z, X_s[-1] * 100, 'g', lw=2)
axes1[0, 1].set_xlabel('z (m)');  axes1[0, 1].set_ylabel('Conversion (%)')
axes1[0, 1].set_title('Monomer conversion')

# Initiator
axes1[0, 2].plot(z, ini_s[-1], 'm', lw=2)
axes1[0, 2].set_xlabel('z (m)');  axes1[0, 2].set_ylabel('[I] (mol/m³)')
axes1[0, 2].set_title('Initiator concentration')

# Mn
axes1[1, 0].plot(z, Mn_s[-1] / 1000, lw=2)
axes1[1, 0].set_xlabel('z (m)');  axes1[1, 0].set_ylabel('Mn (kg/mol)')
axes1[1, 0].set_title('Number-avg molecular weight')

# Mw
axes1[1, 1].plot(z, Mw_s[-1] / 1000, lw=2)
axes1[1, 1].set_xlabel('z (m)');  axes1[1, 1].set_ylabel('Mw (kg/mol)')
axes1[1, 1].set_title('Weight-avg molecular weight')

# PDI
axes1[1, 2].plot(z, PDI_s[-1], lw=2)
axes1[1, 2].set_xlabel('z (m)');  axes1[1, 2].set_ylabel('PDI (—)')
axes1[1, 2].set_title('Polydispersity index')

plt.tight_layout()
plt.savefig('ldpe_pfr_steady_profiles.png', dpi=150)

# %%
# ---- Figure 2: Dynamic evolution at selected axial positions --------------
# %%
z_idx    = [0, N // 5, N // 2, 4 * N // 5, N - 1]
z_labels = [f'z={z[i]:.0f} m' for i in z_idx]
colors   = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle('Dynamic evolution at selected axial positions', fontsize=11)

for idx, lbl, col in zip(z_idx, z_labels, colors):
    axes2[0].plot(sol.t, T_s[:,  idx] - 273.15, color=col, label=lbl)
    axes2[1].plot(sol.t, X_s[:,  idx] * 100,    color=col, label=lbl)
    axes2[2].plot(sol.t, Tc_s[:, idx] - 273.15, color=col, label=lbl)

for ax, title, ylabel in zip(
    axes2,
    ['Reactor T(t)', 'Conversion(t)', 'Coolant Tc(t)'],
    ['T (°C)', 'Conversion (%)', 'Tc (°C)']
):
    ax.set_xlabel('t (s)');  ax.set_ylabel(ylabel)
    ax.set_title(title);     ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('ldpe_pfr_dynamic_evolution.png', dpi=150)

# %%
# ---- Figure 3: 2-D spatio-temporal colormaps ------------------------------
# %%
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4))
fig3.suptitle('Spatio-temporal distribution', fontsize=11)

im0 = axes3[0].contourf(z, sol.t, T_s  - 273.15, levels=30, cmap='hot')
fig3.colorbar(im0, ax=axes3[0], label='T (°C)')
axes3[0].set_xlabel('z (m)');  axes3[0].set_ylabel('t (s)')
axes3[0].set_title('Reactor T(z, t)')

im1 = axes3[1].contourf(z, sol.t, Tc_s - 273.15, levels=30, cmap='cool_r')
fig3.colorbar(im1, ax=axes3[1], label='Tc (°C)')
axes3[1].set_xlabel('z (m)');  axes3[1].set_ylabel('t (s)')
axes3[1].set_title('Coolant Tc(z, t)')

im2 = axes3[2].contourf(z, sol.t, X_s  * 100,     levels=30, cmap='viridis')
fig3.colorbar(im2, ax=axes3[2], label='Conversion (%)')
axes3[2].set_xlabel('z (m)');  axes3[2].set_ylabel('t (s)')
axes3[2].set_title('Conversion X(z, t)')

plt.tight_layout()
plt.savefig('ldpe_pfr_spacetime.png', dpi=150)

plt.show()

# %%
# ---- Print summary --------------------------------------------------------
# %%
print('\n=== Steady-state summary at reactor outlet (z = L) ===')
print(f'  Conversion    : {X_s[-1, -1]*100:.1f} %')
print(f'  T(exit)       : {T_s[-1, -1]-273.15:.1f} °C')
print(f'  Tc(z=0)       : {Tc_s[-1, 0]-273.15:.1f} °C  (coolant outlet)')
print(f'  ΔT max (z)    : {(T_s[-1] - Tc_s[-1]).max():.1f} K')
print(f'  Mn(exit)      : {Mn_s[-1, -1]/1000:.2f} kg/mol')
print(f'  Mw(exit)      : {Mw_s[-1, -1]/1000:.2f} kg/mol')
print(f'  PDI(exit)     : {PDI_s[-1, -1]:.2f}')
