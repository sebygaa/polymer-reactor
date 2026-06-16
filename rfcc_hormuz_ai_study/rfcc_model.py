#!/usr/bin/env python3
"""
rfcc_model.py
Core library for the RFCC (Residue Fluid Catalytic Cracking) crisis-response
study.  Shared by rfcc_step1 ... rfcc_step5.

Physical model
--------------
1) Riser (1-D plug flow, adiabatic, 5-lump kinetics)
     lump 0 : heavy feed / slurry  (HS)   -- unconverted residue + HCO
     lump 1 : LCO                  (light cycle oil, diesel blendstock)
     lump 2 : gasoline             (GASO)
     lump 3 : gas                  (LPG + dry gas)
     lump 4 : coke                 (on catalyst, burned in regenerator)

   Reaction network (feed cracking 2nd order, secondary cracking 1st order):
     HS  -> LCO, GASO, GAS, COKE          (k1l, k1g, k1s, k1c)
     LCO -> GASO, GAS, COKE               (k2g, k2s, k2c)
     GASO-> GAS, COKE   (overcracking)    (k3s, k3c)

   Catalyst deactivation:  phi(t) = exp(-alpha * t),
     alpha = ALPHA0 * (1 + ALPHA_CCR * CCR)     [CCR in wt%]
   Additive (Conradson) coke: deposited instantly at the riser inlet,
     y_coke(0) = F_ADD_COKE * CCR / 100.

2) Riser inlet mix temperature from catalyst / feed enthalpy balance,
   catalyst-to-oil ratio (C/O) solved so that the riser outlet temperature
   (ROT) matches its setpoint -- this mimics the regenerated-catalyst
   slide-valve temperature controller.

3) Regenerator heat balance (full combustion + 3 % losses + catalyst
   cooler) closed iteratively with the riser:  the coke yield sets the
   regenerator temperature, which feeds back into the required C/O.

4) Economics: product values - feed cost - variable opex - catalyst
   make-up (metals/CCR dependent), reported as $/t feed and k$/day.

5) Hormuz-strait crisis scenario generator: 150-day timeline of feed
   quality (CCR), feed availability, feed cost and product prices.

Units: T in K internally (degC at interfaces), energy kJ, mass kg, time s.
"""

import math
import numpy as np

# =============================================================================
# A.  DESIGN BASIS & PHYSICAL CONSTANTS
# =============================================================================
R_GAS   = 8.314          # J/mol/K
T_REF   = 793.15         # K (520 degC) kinetic reference temperature

F_FEED_DESIGN = 84.5     # kg/s  (~50 kBPD residue/VGO blend)
T_RES   = 3.0            # s   riser vapour residence time
N_STEP  = 30             # RK4 steps along the riser

CP_OIL_L  = 2.6          # kJ/kg/K liquid feed
CP_OIL_V  = 3.0          # kJ/kg/K hydrocarbon vapour at riser T
CP_CAT    = 1.09         # kJ/kg/K catalyst
CP_AIR    = 1.07         # kJ/kg/K air at regen T
CP_STEAM  = 2.1          # kJ/kg/K
STEAM_FRAC= 0.05         # kg steam / kg feed (dispersion + lift)
T_STEAM   = 523.15       # K steam supply
DH_VAP    = 250.0        # kJ/kg feed latent heat
T_VAP     = 613.15       # K mean feed boiling (vaporisation) temperature

DH_CRK1 = 500.0          # kJ/kg cracked, feed lump   (endothermic)
DH_CRK2 = 350.0          # kJ/kg cracked, LCO lump
DH_CRK3 = 250.0          # kJ/kg cracked, gasoline lump

DH_COKE = 29_000.0       # kJ/kg coke combustion (CO2 + some CO, H in coke)
Q_LOSS_FRAC = 0.03       # regenerator heat losses
AFR     = 14.0           # kg air / kg coke (full burn + excess O2)
T_AIR   = 473.15         # K main-air-blower discharge

# --- equipment limits (the "hard walls" during a crisis) ---
T_RGN_MAX_C = 730.0                          # degC regenerator metallurgy
Q_COOL_MAX  = 80_000.0                       # kW catalyst-cooler max duty
                                             # (resid FCC dense-bed coolers)
AIR_MAX     = 1.20 * (0.088 * F_FEED_DESIGN * AFR)   # kg/s blower capacity
WGC_MAX     = 1.15 * (0.165 * F_FEED_DESIGN)         # kg/s wet-gas compressor
CO_MIN, CO_MAX = 3.0, 14.0                   # slide-valve C/O range

# --- operating variable bounds (AI action space) ---
ROT_LO, ROT_HI         = 490.0, 545.0        # degC riser outlet T setpoint
PRE_LO, PRE_HI         = 150.0, 330.0        # degC feed preheat
QC_LO,  QC_HI          = 0.0,   1.0          # catalyst cooler duty fraction
RATE_LO, RATE_HI       = 0.55,  1.05         # feed rate fraction of design

# =============================================================================
# B.  5-LUMP KINETICS
# =============================================================================
# k_ref [1/s at C/O = CO_REF, T = T_REF], Ea [J/mol] -- tuned to reproduce
# typical RFCC yields at the heat-balanced base case (ROT 520 degC,
# preheat 220 degC, cooler 50 %, CCR 4 wt%):
#   conv 0.74, gasoline 0.48, coke 0.097, C/O 9.8, T_rgn 668 degC
CO_REF = 6.5
K1G, K1L, K1S, K1C = 1.100, 0.440, 0.280, 0.1200   # HS  -> GASO/LCO/GAS/COKE
K2G, K2S, K2C      = 0.0600, 0.0220, 0.0100        # LCO -> GASO/GAS/COKE
K3S, K3C           = 0.0200, 0.0050                # GASO-> GAS/COKE
E1, E1C = 65e3, 50e3                               # J/mol
E2, E2C = 60e3, 55e3
E3, E3C = 72e3, 68e3

ALPHA0    = 0.12         # 1/s base deactivation
ALPHA_CCR = 0.06         # per wt% CCR (metals + Conradson carbon)
F_ADD_COKE = 0.85        # kg additive coke / kg CCR

_ROR = 1.0 / R_GAS
_ITREF = 1.0 / T_REF


def _rhs(t, y1, y2, y3, T, CO, alpha, cp_tot):
    """Scalar RHS of the riser ODE (only y1..y3 and T drive the rates)."""
    iT = 1.0 / T - _ITREF
    a1 = math.exp(-E1 * _ROR * iT)
    a1c = math.exp(-E1C * _ROR * iT)
    a2 = math.exp(-E2 * _ROR * iT)
    a2c = math.exp(-E2C * _ROR * iT)
    a3 = math.exp(-E3 * _ROR * iT)
    a3c = math.exp(-E3C * _ROR * iT)

    fac = (CO / CO_REF) * math.exp(-alpha * t)
    r1 = fac * y1 * y1          # 2nd order feed cracking
    r2 = fac * y2               # 1st order LCO cracking
    r3 = fac * y3               # 1st order gasoline overcracking

    p1g = K1G * a1 * r1
    p1l = K1L * a1 * r1
    p1s = K1S * a1 * r1
    p1c = K1C * a1c * r1
    p2g = K2G * a2 * r2
    p2s = K2S * a2 * r2
    p2c = K2C * a2c * r2
    p3s = K3S * a3 * r3
    p3c = K3C * a3c * r3

    c1 = p1g + p1l + p1s + p1c
    c2 = p2g + p2s + p2c
    c3 = p3s + p3c
    return (-c1,
            p1l - c2,
            p1g + p2g - c3,
            p1s + p2s + p3s,
            p1c + p2c + p3c,
            -(DH_CRK1 * c1 + DH_CRK2 * c2 + DH_CRK3 * c3) / cp_tot)


def riser(CO, T_mix_K, CCR, t_res=T_RES, n_step=N_STEP, profile=False):
    """Integrate the riser (fixed-step RK4 on scalars, fast).
    Returns final state array [yHS, yLCO, yGASO, yGAS, yCOKE, T(K)] or the
    whole profile (t, Y) when profile=True."""
    alpha = ALPHA0 * (1.0 + ALPHA_CCR * CCR)
    cp_tot = CP_OIL_V + CO * CP_CAT + STEAM_FRAC * CP_STEAM
    add = F_ADD_COKE * CCR / 100.0
    y1, y2, y3, y4, y5, T = 1.0 - add, 0.0, 0.0, 0.0, add, T_mix_K
    h = t_res / n_step
    if profile:
        ts = np.linspace(0.0, t_res, n_step + 1)
        Y = np.zeros((n_step + 1, 6))
        Y[0] = (y1, y2, y3, y4, y5, T)
    t = 0.0
    for i in range(n_step):
        a = _rhs(t, y1, y2, y3, T, CO, alpha, cp_tot)
        b = _rhs(t + 0.5 * h, y1 + 0.5 * h * a[0], y2 + 0.5 * h * a[1],
                 y3 + 0.5 * h * a[2], T + 0.5 * h * a[5], CO, alpha, cp_tot)
        c = _rhs(t + 0.5 * h, y1 + 0.5 * h * b[0], y2 + 0.5 * h * b[1],
                 y3 + 0.5 * h * b[2], T + 0.5 * h * b[5], CO, alpha, cp_tot)
        d = _rhs(t + h, y1 + h * c[0], y2 + h * c[1],
                 y3 + h * c[2], T + h * c[5], CO, alpha, cp_tot)
        h6 = h / 6.0
        y1 += h6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0])
        y2 += h6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1])
        y3 += h6 * (a[2] + 2 * b[2] + 2 * c[2] + d[2])
        y4 += h6 * (a[3] + 2 * b[3] + 2 * c[3] + d[3])
        y5 += h6 * (a[4] + 2 * b[4] + 2 * c[4] + d[4])
        T  += h6 * (a[5] + 2 * b[5] + 2 * c[5] + d[5])
        t += h
        if profile:
            Y[i + 1] = (y1, y2, y3, y4, y5, T)
    if profile:
        return ts, Y
    return np.array([y1, y2, y3, y4, y5, T])


# =============================================================================
# C.  RISER INLET MIX TEMPERATURE & C/O SOLVER
# =============================================================================
def mix_temperature(CO, T_rgn_K, T_pre_K):
    """Riser-bottom mix temperature from catalyst/feed/steam enthalpy
    balance (closed form)."""
    num = (CO * CP_CAT * T_rgn_K
           - CP_OIL_L * (T_VAP - T_pre_K) - DH_VAP + CP_OIL_V * T_VAP
           + STEAM_FRAC * CP_STEAM * T_STEAM)
    den = CO * CP_CAT + CP_OIL_V + STEAM_FRAC * CP_STEAM
    return num / den


def solve_CO(T_rgn_K, ROT_set_K, T_pre_K, CCR, co_guess=None):
    """Find C/O such that riser outlet T = ROT setpoint (slide-valve TC).
    Secant iteration with warm start; g(CO) is monotonically increasing.
    Returns (CO, y_out, flag) -- flag: 0 ok, -1 pinned at CO_MIN (too hot),
    +1 pinned at CO_MAX (cannot reach setpoint)."""
    def gy(CO):
        y = riser(CO, mix_temperature(CO, T_rgn_K, T_pre_K), CCR)
        return y[5] - ROT_set_K, y

    x0 = CO_REF if co_guess is None else min(max(co_guess, CO_MIN), CO_MAX)
    x1 = min(x0 + 0.5, CO_MAX) if x0 < CO_MAX else x0 - 0.5
    g0, y0 = gy(x0)
    g1, y1 = gy(x1)
    for _ in range(20):
        if abs(g1) < 0.05:
            return x1, y1, 0
        if g1 == g0:
            break
        x2 = x1 - g1 * (x1 - x0) / (g1 - g0)
        x2 = min(max(x2, CO_MIN), CO_MAX)
        if x2 == x1:                       # stuck at a bound
            break
        x0, g0 = x1, g1
        x1 = x2
        g1, y1 = gy(x1)
    # secant pinned at a bound or failed -> check bounds / fall back
    g_lo, y_lo = gy(CO_MIN)
    if g_lo >= 0.0:
        return CO_MIN, y_lo, -1
    g_hi, y_hi = gy(CO_MAX)
    if g_hi <= 0.0:
        return CO_MAX, y_hi, +1
    lo, hi = CO_MIN, CO_MAX                # robust bisection fallback
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        gm, ym = gy(mid)
        if abs(gm) < 0.05 or hi - lo < 1e-3:
            return mid, ym, 0
        if gm > 0.0:
            hi = mid
        else:
            lo = mid
    return mid, ym, 0


# =============================================================================
# D.  INTEGRATED UNIT (RISER + REGENERATOR HEAT BALANCE)
# =============================================================================
def solve_unit(ROT_C, T_pre_C, Qcool_frac, CCR, feed_frac=1.0,
               T_rgn_init_C=690.0, tol=0.05, max_iter=120):
    """Solve the heat-balanced riser/regenerator at one operating point.

    Decision variables : ROT_C [degC], T_pre_C [degC], Qcool_frac [0-1],
                         feed_frac (of design rate)
    Disturbance        : CCR [wt%] of the feed
    Returns a result dict (yields, C/O, T_rgn, flows, constraint slacks).
    """
    ROT_K   = ROT_C + 273.15
    T_pre_K = T_pre_C + 273.15
    F_feed  = F_FEED_DESIGN * max(feed_frac, 1e-3)
    q_cool  = Qcool_frac * Q_COOL_MAX / F_feed          # kJ/kg feed

    T_rgn = T_rgn_init_C + 273.15
    CO, y, flag = CO_REF, None, 0
    co_guess = None
    for _ in range(max_iter):
        CO, y, flag = solve_CO(T_rgn, ROT_K, T_pre_K, CCR, co_guess)
        co_guess = CO
        y_coke = y[4]
        T_out_K = y[5]
        q_rel = y_coke * DH_COKE * (1.0 - Q_LOSS_FRAC) - q_cool
        a_cat = CO * CP_CAT
        a_air = y_coke * AFR * CP_AIR
        T_new = (q_rel + a_cat * T_out_K + a_air * T_AIR) / (a_cat + a_air)
        T_new = max(T_new, 850.0)                       # regen won't sustain
        if abs(T_new - T_rgn) < tol:
            T_rgn = T_new
            break
        T_rgn += 0.5 * (T_new - T_rgn)

    yields = y[:5].copy()
    T_out_C = y[5] - 273.15
    conv = 1.0 - yields[0] - yields[1]
    F_air = yields[4] * AFR * F_feed
    F_wet_gas = yields[3] * F_feed

    # dry-gas fraction of the gas lump rises with riser temperature (thermal)
    f_dry = float(np.clip(0.15 + 0.004 * (T_out_C - 500.0), 0.08, 0.45))

    return dict(
        ROT_set=ROT_C, T_pre=T_pre_C, Qcool_frac=Qcool_frac,
        feed_frac=feed_frac, CCR=CCR,
        y_hs=yields[0], y_lco=yields[1], y_gaso=yields[2],
        y_gas=yields[3], y_coke=yields[4],
        f_dry=f_dry, conv=conv, CO=CO, co_flag=flag,
        T_rgn_C=T_rgn - 273.15, T_out_C=T_out_C,
        T_mix_C=mix_temperature(CO, T_rgn, T_pre_K) - 273.15,
        F_feed=F_feed, F_air=F_air, F_wet_gas=F_wet_gas,
        viol_Trgn=max(0.0, (T_rgn - 273.15) - T_RGN_MAX_C),
        viol_air=max(0.0, F_air / AIR_MAX - 1.0),
        viol_wgc=max(0.0, F_wet_gas / WGC_MAX - 1.0),
    )


# =============================================================================
# E.  ECONOMICS
# =============================================================================
PRICES_BASE = dict(feed=460.0, gaso=720.0, lco=640.0, lpg=520.0,
                   drygas=180.0, slurry=380.0)        # $/t
OPEX_VAR  = 15.0          # $/t feed (utilities, steam, power)
CAT_BASE  = 2.0           # $/t feed catalyst make-up at CCR = 0
CAT_CCR   = 0.12          # relative increase per wt% CCR (metals poisoning)
T_PER_DAY = F_FEED_DESIGN * 86_400.0 / 1000.0         # t/day at design


def economics(res, prices=None):
    """Margin per tonne of feed and k$/day from a solve_unit result."""
    p = PRICES_BASE if prices is None else prices
    y_lpg = res['y_gas'] * (1.0 - res['f_dry'])
    y_dry = res['y_gas'] * res['f_dry']
    revenue = (res['y_gaso'] * p['gaso'] + res['y_lco'] * p['lco']
               + y_lpg * p['lpg'] + y_dry * p['drygas']
               + res['y_hs'] * p['slurry'])
    cat_cost = CAT_BASE * (1.0 + CAT_CCR * res['CCR'])
    margin_t = revenue - p['feed'] - OPEX_VAR - cat_cost   # $/t feed
    tpd = T_PER_DAY * res['feed_frac']
    return dict(margin_per_t=margin_t,
                margin_kUSD_day=margin_t * tpd / 1000.0,
                revenue_per_t=revenue, tpd=tpd)


# =============================================================================
# F.  HORMUZ-STRAIT CRISIS SCENARIO GENERATOR
# =============================================================================
def _ramp(d, d0, d1, v0, v1):
    """Linear ramp helper."""
    if d <= d0:
        return v0
    if d >= d1:
        return v1
    return v0 + (v1 - v0) * (d - d0) / (d1 - d0)


def scenario_timeline(n_days=150, seed=42, noise=True):
    """Daily crisis timeline.

    Phase 0  d <  15 : normal -- Middle-East VGO/AR blend (CCR 4 wt%)
    Phase 1  15- 35  : BLOCKADE SHOCK -- ME cargoes stop; refinery draws
                       stored heavy domestic atmospheric residue
                       (CCR -> 6.8), supply 100->65 %, feed cost x1.75,
                       gasoline/diesel cracks blow out (Gulf product
                       exports are cut too, so cracks rise MORE than crude)
    Phase 2  35- 70  : ADAPTATION -- expensive light-sweet alternatives
                       (USGC/WAF) arrive: CCR -> 2.5, supply -> 85 %,
                       feed cost stays x1.6 (freight premium)
    Phase 3  70-110  : PARTIAL REOPEN -- escorted convoys, blend returns
                       toward ME quality (CCR 4.5), supply 95 %, x1.25
    Phase 4  d >=110 : normalisation
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)
    CCR = np.zeros(n_days); sup = np.zeros(n_days); cf = np.zeros(n_days)
    pg = np.zeros(n_days); pl = np.zeros(n_days); ps = np.zeros(n_days)
    plpg = np.zeros(n_days); phase = np.zeros(n_days, dtype=int)

    for d in days:
        if d < 15:
            ph = 0
            ccr, s, f = 4.0, 1.0, 1.0
            g, l, sl, lp = 1.0, 1.0, 1.0, 1.0
        elif d < 35:
            ph = 1
            ccr = _ramp(d, 15, 22, 4.0, 6.8)
            s   = _ramp(d, 15, 20, 1.0, 0.65)
            f   = _ramp(d, 15, 19, 1.0, 1.75)
            g   = _ramp(d, 15, 20, 1.0, 1.55)
            l   = _ramp(d, 15, 20, 1.0, 1.70)
            sl  = _ramp(d, 15, 20, 1.0, 1.30)
            lp  = _ramp(d, 15, 20, 1.0, 1.25)
        elif d < 70:
            ph = 2
            ccr = _ramp(d, 35, 48, 6.8, 2.5)
            s   = _ramp(d, 35, 50, 0.65, 0.85)
            f   = _ramp(d, 35, 45, 1.75, 1.60)
            g   = _ramp(d, 35, 55, 1.55, 1.45)
            l   = _ramp(d, 35, 55, 1.70, 1.55)
            sl, lp = 1.30, 1.25
        elif d < 110:
            ph = 3
            ccr = _ramp(d, 70, 90, 2.5, 4.5)
            s   = _ramp(d, 70, 85, 0.85, 0.95)
            f   = _ramp(d, 70, 95, 1.60, 1.25)
            g   = _ramp(d, 70, 95, 1.45, 1.12)
            l   = _ramp(d, 70, 95, 1.55, 1.18)
            sl  = _ramp(d, 70, 95, 1.30, 1.10)
            lp  = _ramp(d, 70, 95, 1.20, 1.08)
        else:
            ph = 4
            ccr = _ramp(d, 110, 130, 4.5, 4.0)
            s   = _ramp(d, 110, 120, 0.95, 1.0)
            f   = _ramp(d, 110, 135, 1.25, 1.02)
            g   = _ramp(d, 110, 135, 1.12, 1.0)
            l   = _ramp(d, 110, 135, 1.18, 1.0)
            sl  = _ramp(d, 110, 135, 1.10, 1.0)
            lp  = _ramp(d, 110, 135, 1.08, 1.0)
        CCR[d], sup[d], cf[d] = ccr, s, f
        pg[d], pl[d], ps[d], plpg[d] = g, l, sl, lp
        phase[d] = ph

    if noise:
        def ar1(sig, rho=0.7):
            e = np.zeros(n_days)
            for i in range(1, n_days):
                e[i] = rho * e[i - 1] + rng.normal(0, sig)
            return e
        CCR = np.clip(CCR + ar1(0.12), 0.5, 9.0)
        sup = np.clip(sup + ar1(0.012), 0.40, 1.0)
        cf  = np.clip(cf + ar1(0.02), 0.8, 2.2)
        pg  = np.clip(pg + ar1(0.02), 0.8, 1.8)
        pl  = np.clip(pl + ar1(0.02), 0.8, 1.9)

    b = PRICES_BASE
    return dict(
        days=days, phase=phase, CCR=CCR, supply=sup,
        p_feed=b['feed'] * cf, p_gaso=b['gaso'] * pg, p_lco=b['lco'] * pl,
        p_lpg=b['lpg'] * plpg, p_drygas=np.full(n_days, b['drygas']),
        p_slurry=b['slurry'] * ps,
    )


def prices_of_day(scen, d):
    return dict(feed=scen['p_feed'][d], gaso=scen['p_gaso'][d],
                lco=scen['p_lco'][d], lpg=scen['p_lpg'][d],
                drygas=scen['p_drygas'][d], slurry=scen['p_slurry'][d])


# =============================================================================
# G.  BASELINE ("design-book") OPERATING POLICY WITH EMERGENCY TRIMS
# =============================================================================
OPS_DESIGN = dict(ROT_C=520.0, T_pre_C=220.0, Qcool_frac=0.5)


def _operator_trims(ops, CCR):
    """Reactive emergency steps an operator would take to stay inside
    equipment limits (applied to baseline AND on top of the AI policy):
       hot regen  -> cat cooler up, then ROT down 3 degC steps
       cold regen (C/O pinned at max, ROT unreachable)
                  -> cat cooler down, then feed preheat up 10 degC steps
       blower / WGC limited -> cut feed rate 2 % steps."""
    ops = dict(ops)
    res = solve_unit(ops['ROT_C'], ops['T_pre_C'], ops['Qcool_frac'],
                     CCR, ops['feed_frac'])
    n = 0
    while res['viol_Trgn'] > 0 and n < 30:
        if ops['Qcool_frac'] < 1.0:
            ops['Qcool_frac'] = min(1.0, ops['Qcool_frac'] + 0.25)
        elif ops['ROT_C'] > ROT_LO:
            ops['ROT_C'] = max(ROT_LO, ops['ROT_C'] - 3.0)
        else:
            break
        res = solve_unit(ops['ROT_C'], ops['T_pre_C'], ops['Qcool_frac'],
                         CCR, ops['feed_frac'])
        n += 1
    while res['co_flag'] == 1 and n < 50:        # cold side
        if ops['Qcool_frac'] > 0.0:
            ops['Qcool_frac'] = max(0.0, ops['Qcool_frac'] - 0.25)
        elif ops['T_pre_C'] < PRE_HI:
            ops['T_pre_C'] = min(PRE_HI, ops['T_pre_C'] + 10.0)
        else:
            break
        res = solve_unit(ops['ROT_C'], ops['T_pre_C'], ops['Qcool_frac'],
                         CCR, ops['feed_frac'])
        n += 1
    while (res['viol_air'] > 0 or res['viol_wgc'] > 0) and \
            ops['feed_frac'] > RATE_LO and n < 80:
        ops['feed_frac'] = max(RATE_LO, ops['feed_frac'] - 0.02)
        res = solve_unit(ops['ROT_C'], ops['T_pre_C'], ops['Qcool_frac'],
                         CCR, ops['feed_frac'])
        n += 1
    return ops, res, n


def baseline_day(CCR, supply, prices):
    """Fixed design-book operation + reactive emergency trims."""
    ops = dict(OPS_DESIGN)
    ops['feed_frac'] = min(1.0, supply)
    ops, res, _ = _operator_trims(ops, CCR)
    eco = economics(res, prices)
    return ops, res, eco


def safety_trim(ops, CCR, prices):
    """Same emergency logic applied to ANY proposed ops (safety layer used
    on top of the AI policy at deployment, for a fair comparison)."""
    ops, res, _ = _operator_trims(ops, CCR)
    eco = economics(res, prices)
    return ops, res, eco


# =============================================================================
# H.  QUICK SELF-TEST
# =============================================================================
if __name__ == '__main__':
    import time
    t0 = time.time()
    res = solve_unit(520.0, 220.0, 0.5, CCR=4.0)
    eco = economics(res)
    dt = time.time() - t0
    print(f"Base case solved in {dt*1e3:.1f} ms")
    for k in ('conv', 'y_gaso', 'y_lco', 'y_gas', 'y_coke', 'y_hs',
              'CO', 'T_rgn_C', 'T_mix_C', 'T_out_C'):
        print(f"  {k:10s} = {res[k]:.4f}")
    print(f"  margin     = {eco['margin_per_t']:.2f} $/t "
          f"({eco['margin_kUSD_day']:.0f} k$/day)")
