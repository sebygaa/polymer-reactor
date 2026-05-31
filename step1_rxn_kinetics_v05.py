# %%
# Importing Packages
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# LDPE Free-Radical Polymerization — Method of Moments + Energy Balance
# Version 05: Physically realistic cooling parameters
#
# Changes from v04:
#   1. D      : 0.05 → 0.02 m   (벽면적/부피 비 6.25배 증가 → 냉각 강화)
#   2. U_heat : 400  → 2000 W/(m²·K)  (냉각 성능 5배 향상)
#   3. Tc_in  : 140  → 60 °C    (냉각 구동력 대폭 증가; 가압 온수 냉각)
#   4. tau_c  : 30   → 20 s     (냉각수 순환 빠르게)
#   5. ini_0  : 100  → 0.3 mol/m³  (실제 LDPE 개시제 농도 수준)
#   6. t_end  : 20   → 300 s    (정상상태 도달까지 충분한 시간)
#
# 개선 근거:
#   - 단열 온도상승 ΔTad = (-ΔHp·M₀)/(ρ·Cp) = 93000·20050/(600000·1.7) = 1828 K
#     → 완전 전환 시 이론 온도상승이 매우 큼
#   - 냉각 계수 h_r = 4U/(ρ·Cp·D):
#       v04: 4·400/(600000·1.7·0.05)   = 0.031 s⁻¹
#       v05: 4·2000/(600000·1.7·0.02)  = 0.392 s⁻¹  (12.6배 향상)
#   - ini_0 감소로 라디칼 생성 속도를 제한하여 급격한 열폭주 억제
#
# 목표 운전 조건:
#   T_peak ≈ 200–270 °C  (실제 LDPE: 150–300 °C)
#   Conversion ≈ 15–35 %
#   PDI ≈ 2–10
# =============================================================================

# %%
# ---- Physical / Process Constants ----------------------------------------
# %%
R_gas   = 8.3145          # J/(mol·K)
Mw_mono = 28.054          # g/mol
f_eff   = 0.8

# Reactor geometry — 소경관 (소형 직경) for high wall-area/volume ratio
D       = 0.02            # m  (v04: 0.05 m)
U_heat  = 2000.0          # W/(m²·K)  (v04: 400)

rho     = 600_000.0       # g/m³
Cp      = 1.7             # J/(g·K)
dH_p    = -93_000.0       # J/mol

# %%
# ---- Cooling Jacket Parameters -------------------------------------------
# %%
# 가압 온수 냉각 (pressurised hot-water jacket, ~10 bar)
# Tc_in = 60 °C: 반응기 공급온도(150 °C)보다 충분히 낮아 냉각 구동력 확보
Tc_in   = 60.0 + 273.15   # K  (v04: 140 °C)
tau_c   = 20.0             # s  (v04: 30 s)
rho_c   = 900_000.0        # g/m³
Cp_c    = 4.18             # J/(g·K)
beta_j  = 0.5              # Vj/Vr

alpha_j = 4.0 * U_heat / (D * rho_c * Cp_c * beta_j)

# 냉각 계수 비교 출력
h_r_v04 = 4.0 * 400   / (600_000.0 * 1.7 * 0.05)
h_r_v05 = 4.0 * U_heat / (rho * Cp * D)
print(f'h_r (v04): {h_r_v04:.4f} s⁻¹')
print(f'h_r (v05): {h_r_v05:.4f} s⁻¹  ({h_r_v05/h_r_v04:.1f}x improvement)')
print(f'alpha_j  : {alpha_j:.4f} s⁻¹')

# %%
# ---- Arrhenius Parameters ------------------------------------------------
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
# ---- ODE System -----------------------------------------------------------
# %%
def rxn_odes(t, y):
    lamb0, lamb1, lamb2 = y[0], y[1], y[2]
    mu0,   mu1,   mu2   = y[3], y[4], y[5]
    ini,   mono,  T, Tc = y[6], y[7], y[8], y[9]

    kd   = arrhenius(A_kd,   Ea_kd,   T)
    kp   = arrhenius(A_kp,   Ea_kp,   T)
    ktc  = arrhenius(A_ktc,  Ea_ktc,  T)
    ktd  = arrhenius(A_ktd,  Ea_ktd,  T)
    ktrm = arrhenius(A_ktrm, Ea_ktrm, T)
    ktrp = arrhenius(A_ktrp, Ea_ktrp, T)

    eps = 1e-12
    if mu0 < eps or mu1 < eps:
        mu3 = 0.0
    else:
        mu3 = mu2 * (2.0 * mu0 * mu2 - mu1**2) / (mu0 * mu1)
    mu3_cs = mu2**2 / (mu1 + eps)        # Cauchy-Schwarz lower bound
    mu3    = max(mu3, mu3_cs, 0.0)

    dini_dt  = -kd * ini
    dmono_dt = -kp * mono * lamb0

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

    dmu0_dt = (ktrm * mono * lamb0
               + (0.5 * ktc + ktd) * lamb0**2)

    dmu1_dt = (ktrm * mono * lamb1
               + ktrp * (lamb1 * mu1 - lamb0 * mu2)
               + (ktc + ktd) * lamb0 * lamb1)

    dmu2_dt = (ktrm * mono * lamb2
               + ktd  * lamb0 * lamb2
               + ktc  * (lamb0 * lamb2 + lamb1**2)
               + ktrp * (lamb2 * mu1 - lamb0 * mu3))

    dT_dt = ((-dH_p) / (rho * Cp) * kp * mono * lamb0
             - 4.0 * U_heat / (rho * Cp * D) * (T - Tc))

    dTc_dt = ((Tc_in - Tc) / tau_c
              + alpha_j * (T - Tc))

    return [dlamb0_dt, dlamb1_dt, dlamb2_dt,
            dmu0_dt,   dmu1_dt,   dmu2_dt,
            dini_dt,   dmono_dt,  dT_dt, dTc_dt]


# %%
# ---- Initial Conditions ---------------------------------------------------
# %%
mono_0 = 2.005e4
ini_0  = 0.3             # mol/m³  (v04: 100)
T_0    = 150 + 273.15
Tc_0   = Tc_in

y0 = np.zeros(10)
y0[6] = ini_0
y0[7] = mono_0
y0[8] = T_0
y0[9] = Tc_0

# %%
# ---- Numerical Integration ------------------------------------------------
# %%
t_end  = 300.0           # s  (v04: 20 s)
t_eval = np.linspace(0.0, t_end, 6000)

sol = solve_ivp(rxn_odes, (0.0, t_end), y0,
                method='Radau',
                t_eval=t_eval,
                rtol=1e-6, atol=1e-10,
                dense_output=False)

t_res = sol.t
y_res = sol.y.T
print(f'Integration: status={sol.status}, nfev={sol.nfev}')

# %%
# ---- Post-processing ------------------------------------------------------
# %%
lamb0_res = y_res[:, 0];  lamb1_res = y_res[:, 1];  lamb2_res = y_res[:, 2]
mu0_res   = y_res[:, 3];  mu1_res   = y_res[:, 4];  mu2_res   = y_res[:, 5]
ini_res   = y_res[:, 6];  mono_res  = y_res[:, 7]
T_res     = y_res[:, 8];  Tc_res    = y_res[:, 9]

eps = 1e-30
# Dead-polymer only Mn/Mw (물리적으로 올바른 고분자 특성화)
Mn_dead  = Mw_mono * (mu1_res + eps) / (mu0_res + eps)
Mw_dead  = Mw_mono * (mu2_res + eps) / (mu1_res + eps)
PDI_dead = Mw_dead / Mn_dead

conversion = 1.0 - mono_res / mono_0

# %%
# ---- Plots ----------------------------------------------------------------
# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle(
    f'LDPE MoM Simulation (v05) — D={D*100:.0f}cm, U={U_heat:.0f}W/m²K, '
    f'Tc_in={Tc_in-273.15:.0f}°C, ini₀={ini_0}mol/m³',
    fontsize=10
)

axes[0, 0].plot(t_res, mono_res / mono_0 * 100)
axes[0, 0].set_ylabel('Monomer remaining (%)'); axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_title('Monomer conversion')

axes[0, 1].plot(t_res, T_res  - 273.15, color='tomato',    label='Reactor T')
axes[0, 1].plot(t_res, Tc_res - 273.15, color='steelblue', label='Coolant Tc')
axes[0, 1].axhline(Tc_in - 273.15, color='steelblue', linestyle='--', alpha=0.5, label='Tc_in')
axes[0, 1].set_ylabel('Temperature (°C)'); axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_title('Reactor & coolant temperature'); axes[0, 1].legend(fontsize=8)

axes[0, 2].plot(t_res, T_res - Tc_res, color='darkorange')
axes[0, 2].set_ylabel('T − Tc  (K)'); axes[0, 2].set_xlabel('Time (s)')
axes[0, 2].set_title('Temperature driving force')

axes[0, 3].plot(t_res, ini_res)
axes[0, 3].set_ylabel('[I] (mol/m³)'); axes[0, 3].set_xlabel('Time (s)')
axes[0, 3].set_title('Initiator concentration')

axes[1, 0].plot(t_res, Mn_dead / 1000)
axes[1, 0].set_ylabel('Mn (kg/mol)'); axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_title('Number-avg mol. weight (dead)')

axes[1, 1].plot(t_res, Mw_dead / 1000)
axes[1, 1].set_ylabel('Mw (kg/mol)'); axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_title('Weight-avg mol. weight (dead)')

axes[1, 2].plot(t_res, PDI_dead)
axes[1, 2].set_ylabel('PDI (—)'); axes[1, 2].set_xlabel('Time (s)')
axes[1, 2].set_title('Polydispersity index (dead)')

axes[1, 3].plot(t_res, conversion * 100)
axes[1, 3].set_ylabel('Conversion (%)'); axes[1, 3].set_xlabel('Time (s)')
axes[1, 3].set_title('Monomer conversion')

plt.tight_layout()
plt.savefig('ldpe_simulation_v05.png', dpi=150)
plt.show()

# %%
# ---- Print summary --------------------------------------------------------
# %%
idx_peak = np.argmax(T_res)
print('\n=== Simulation Summary ===')
print(f'  T_peak             : {T_res[idx_peak]-273.15:.1f} °C  at t={t_res[idx_peak]:.1f} s')
print(f'  T_final            : {T_res[-1]-273.15:.1f} °C')
print(f'  Tc_final           : {Tc_res[-1]-273.15:.1f} °C')
print(f'  T−Tc (final)       : {T_res[-1]-Tc_res[-1]:.1f} K')
print(f'  Monomer conversion : {conversion[-1]*100:.1f} %')
print(f'  Mn (dead, final)   : {Mn_dead[-1]/1000:.2f} kg/mol')
print(f'  Mw (dead, final)   : {Mw_dead[-1]/1000:.2f} kg/mol')
print(f'  PDI (dead, final)  : {PDI_dead[-1]:.2f}')
print(f'  ΔTad (full conv.)  : {(-dH_p)*mono_0/(rho*Cp):.0f} K  (reference)')
