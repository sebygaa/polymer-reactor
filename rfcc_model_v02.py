#!/usr/bin/env python3
"""
rfcc_model_v02.py
RFCC crisis-response study -- model v02.

Changes from rfcc_model.py (v01):

  v02-1  The unit is restructured into two explicit, separately audited
         vessels coupled only by the catalyst circulation streams:

           RiserReactor   regenerated cat (T_rgn) + feed + steam
                          -> products + spent cat (T_out, coke on cat)
           Regenerator    spent cat + air -> flue gas + regenerated cat
                          (+ catalyst cooler + wall losses)

  v02-2  Explicit open-system ENERGY BALANCES on both vessels, written as
         enthalpy streams vs a common reference state T0 = 298.15 K.

         Riser reactor (per kg feed, kJ/kg):
           CO*cp_cat*(T_rgn-T0) + cp_oil_l*(T_pre-T0)
             + S*cp_steam*(T_stm-T0)                              [IN]
           = CO*cp_cat*(T_out-T0)
             + [cp_oil_l*(Tvap-T0) + dHvap + cp_oil_v*(T_out-Tvap)]
             + S*cp_steam*(T_out-T0) + Q_rxn                      [OUT]
         Q_rxn is rebuilt from the integrated amounts cracked per lump
         (dH_i * m_cracked_i, states 7-8).  Because mass and energy are
         integrated by the same RK4, closure is an exact identity of the
         integrator: the residual audits the BOOKKEEPING (mix-T formula,
         ODE energy term, enthalpy-stream definitions staying mutually
         consistent), while discretisation accuracy is verified
         separately by step refinement (Step 6d).

  v02-3  Regenerator combustion from coke STOICHIOMETRY instead of the
         fixed AFR / lumped dH_coke of v01:
           coke = CH_n (n = H/C atomic ratio), carbon split x_CO2 to CO2
           and (1-x_CO2) to CO, hydrogen to H2O; air from O2 demand plus
           excess; heat release from formation enthalpies; flue gas
           leaves at T_rgn with its own cp.
         Calibrated (x_CO2 = 0.70, 15 % excess air) so the v02 base case
         reproduces the v01 heat-balanced base case within a few degC.

  v02-4  Every solve returns the full energy-balance term breakdown and
         closure residuals for both vessels (res['eb_reactor'],
         res['eb_regen']), enabling the Step-6 verification study.

Interface is kept compatible with v01: solve_unit / baseline_day /
safety_trim have the same signatures; economics & scenarios are imported
unchanged from rfcc_model.
"""

import math
import numpy as np

# physical constants, kinetics, bounds and economics shared with v01
from rfcc_model import (
    R_GAS, T_REF, F_FEED_DESIGN, T_RES, N_STEP,
    CP_OIL_L, CP_OIL_V, CP_CAT, CP_AIR, CP_STEAM,
    STEAM_FRAC, T_STEAM, DH_VAP, T_VAP,
    DH_CRK1, DH_CRK2, DH_CRK3, Q_LOSS_FRAC, T_AIR,
    T_RGN_MAX_C, Q_COOL_MAX, AIR_MAX, WGC_MAX, CO_MIN, CO_MAX,
    ROT_LO, ROT_HI, PRE_LO, PRE_HI, RATE_LO, RATE_HI,
    K1G, K1L, K1S, K1C, K2G, K2S, K2C, K3S, K3C,
    E1, E1C, E2, E2C, E3, E3C, ALPHA0, ALPHA_CCR, F_ADD_COKE,
    CO_REF, _ROR, _ITREF, OPS_DESIGN,
    economics, scenario_timeline, prices_of_day, PRICES_BASE,
)

T0_REF = 298.15          # K  common enthalpy reference state
CP_COKE = 1.10           # kJ/kg/K coke on catalyst (solid)
CP_FLUE = 1.12           # kJ/kg/K flue gas at regenerator T

# --- coke combustion stoichiometry (v02-3) ---
# Partial-burn split (x_CO2 = 0.60) is typical for resid FCC regenerators
# operated CO-rich to limit the bed temperature (CO boiler downstream);
# calibrated together with the excess air so the v02 base case matches
# the v01 lumped heat balance (T_rgn 668 degC, C/O 9.8, conv 0.74).
COKE_H_C = 0.70          # atomic H/C ratio of coke (CH_0.7)
X_CO2 = 0.60             # fraction of coke carbon burned to CO2
EXCESS_AIR = 0.15        # fractional excess air over stoichiometric
MW_C, MW_H, MW_AIR = 12.011, 1.008, 28.96
DHF_CO2, DHF_CO, DHF_H2O = 393.5e3, 110.5e3, 241.8e3   # kJ/kmol released

# per kg coke: mass split and mole numbers
_MW_COKE = MW_C + COKE_H_C * MW_H
W_C = MW_C / _MW_COKE                       # kg C / kg coke
W_H = COKE_H_C * MW_H / _MW_COKE            # kg H / kg coke
N_C = W_C / MW_C                            # kmol C / kg coke
N_H2 = W_H / (2.0 * MW_H)                   # kmol H2 / kg coke

# O2 demand, air and flue mass per kg coke
N_O2 = N_C * (X_CO2 + 0.5 * (1.0 - X_CO2)) + 0.5 * N_H2
AFR_STOICH = N_O2 / 0.21 * MW_AIR
AFR_V02 = AFR_STOICH * (1.0 + EXCESS_AIR)   # kg air / kg coke
# combustion heat release per kg coke [kJ/kg]
DH_COKE_V02 = (N_C * (X_CO2 * DHF_CO2 + (1.0 - X_CO2) * DHF_CO)
               + N_H2 * DHF_H2O)

# main-air-blower capacity restated on the v02 stoichiometric air basis
# (same design margin as v01: 120 % of design-coke air demand)
AIR_MAX_V02 = 1.20 * (0.088 * F_FEED_DESIGN * AFR_V02)


# =============================================================================
# A.  RISER REACTOR (vessel 1)
# =============================================================================
class RiserReactor:
    """1-D adiabatic plug-flow riser with 5-lump kinetics (v01 chemistry)
    plus independent integration of the heat absorbed per cracking path
    (states 7-8: mass cracked from the LCO and gasoline lumps)."""

    def __init__(self, t_res=T_RES, n_step=N_STEP):
        self.t_res = t_res
        self.n_step = n_step

    @staticmethod
    def _rhs(t, y1, y2, y3, T, CO, alpha, cp_tot):
        """Returns d/dt of (y1, y2, y3, y4, y5, T, cr2, cr3)."""
        iT = 1.0 / T - _ITREF
        a1 = math.exp(-E1 * _ROR * iT)
        a1c = math.exp(-E1C * _ROR * iT)
        a2 = math.exp(-E2 * _ROR * iT)
        a2c = math.exp(-E2C * _ROR * iT)
        a3 = math.exp(-E3 * _ROR * iT)
        a3c = math.exp(-E3C * _ROR * iT)

        fac = (CO / CO_REF) * math.exp(-alpha * t)
        r1 = fac * y1 * y1
        r2 = fac * y2
        r3 = fac * y3

        p1g = K1G * a1 * r1; p1l = K1L * a1 * r1
        p1s = K1S * a1 * r1; p1c = K1C * a1c * r1
        p2g = K2G * a2 * r2; p2s = K2S * a2 * r2; p2c = K2C * a2c * r2
        p3s = K3S * a3 * r3; p3c = K3C * a3c * r3

        c1 = p1g + p1l + p1s + p1c
        c2 = p2g + p2s + p2c
        c3 = p3s + p3c
        return (-c1,
                p1l - c2,
                p1g + p2g - c3,
                p1s + p2s + p3s,
                p1c + p2c + p3c,
                -(DH_CRK1 * c1 + DH_CRK2 * c2 + DH_CRK3 * c3) / cp_tot,
                c2,
                c3)

    def mix_temperature(self, CO, T_rgn_K, T_pre_K):
        """Riser-bottom mix T from the vessel inlet enthalpy balance."""
        num = (CO * CP_CAT * T_rgn_K
               - CP_OIL_L * (T_VAP - T_pre_K) - DH_VAP + CP_OIL_V * T_VAP
               + STEAM_FRAC * CP_STEAM * T_STEAM)
        den = CO * CP_CAT + CP_OIL_V + STEAM_FRAC * CP_STEAM
        return num / den

    def solve(self, CO, T_rgn_K, T_pre_K, CCR, profile=False):
        """Integrate the riser (RK4).  Returns dict with yields, T_out,
        T_mix and the independently integrated cracking duty q_rxn."""
        alpha = ALPHA0 * (1.0 + ALPHA_CCR * CCR)
        cp_tot = CP_OIL_V + CO * CP_CAT + STEAM_FRAC * CP_STEAM
        add = F_ADD_COKE * CCR / 100.0
        T_mix = self.mix_temperature(CO, T_rgn_K, T_pre_K)
        y1, y2, y3, y4, y5 = 1.0 - add, 0.0, 0.0, 0.0, add
        T, cr2, cr3 = T_mix, 0.0, 0.0
        h = self.t_res / self.n_step
        if profile:
            ts = np.linspace(0.0, self.t_res, self.n_step + 1)
            Y = np.zeros((self.n_step + 1, 8))
            Y[0] = (y1, y2, y3, y4, y5, T, cr2, cr3)
        t = 0.0
        for i in range(self.n_step):
            a = self._rhs(t, y1, y2, y3, T, CO, alpha, cp_tot)
            b = self._rhs(t + 0.5 * h, y1 + 0.5 * h * a[0],
                          y2 + 0.5 * h * a[1], y3 + 0.5 * h * a[2],
                          T + 0.5 * h * a[5], CO, alpha, cp_tot)
            c = self._rhs(t + 0.5 * h, y1 + 0.5 * h * b[0],
                          y2 + 0.5 * h * b[1], y3 + 0.5 * h * b[2],
                          T + 0.5 * h * b[5], CO, alpha, cp_tot)
            d = self._rhs(t + h, y1 + h * c[0], y2 + h * c[1],
                          y3 + h * c[2], T + h * c[5], CO, alpha, cp_tot)
            h6 = h / 6.0
            y1 += h6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0])
            y2 += h6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1])
            y3 += h6 * (a[2] + 2 * b[2] + 2 * c[2] + d[2])
            y4 += h6 * (a[3] + 2 * b[3] + 2 * c[3] + d[3])
            y5 += h6 * (a[4] + 2 * b[4] + 2 * c[4] + d[4])
            T  += h6 * (a[5] + 2 * b[5] + 2 * c[5] + d[5])
            cr2 += h6 * (a[6] + 2 * b[6] + 2 * c[6] + d[6])
            cr3 += h6 * (a[7] + 2 * b[7] + 2 * c[7] + d[7])
            t += h
            if profile:
                Y[i + 1] = (y1, y2, y3, y4, y5, T, cr2, cr3)
        cracked1 = (1.0 - add) - y1
        q_rxn = DH_CRK1 * cracked1 + DH_CRK2 * cr2 + DH_CRK3 * cr3
        out = dict(yields=np.array([y1, y2, y3, y4, y5]),
                   T_out_K=T, T_mix_K=T_mix, q_rxn=q_rxn,
                   cracked=(cracked1, cr2, cr3), CO=CO, CCR=CCR,
                   T_rgn_K=T_rgn_K, T_pre_K=T_pre_K)
        if profile:
            out['profile'] = (ts, Y)
        return out

    @staticmethod
    def energy_balance(r):
        """Enthalpy-stream audit of one riser solution (kJ/kg feed).
        Residual = IN - OUT; q_rxn comes from the cracked-mass integrals,
        so a near-zero residual verifies the energy ODE numerically."""
        CO = r['CO']
        q_cat_in = CO * CP_CAT * (r['T_rgn_K'] - T0_REF)
        q_feed_in = CP_OIL_L * (r['T_pre_K'] - T0_REF)
        q_steam_in = STEAM_FRAC * CP_STEAM * (T_STEAM - T0_REF)
        q_cat_out = CO * CP_CAT * (r['T_out_K'] - T0_REF)
        q_oil_out = (CP_OIL_L * (T_VAP - T0_REF) + DH_VAP
                     + CP_OIL_V * (r['T_out_K'] - T_VAP))
        q_steam_out = STEAM_FRAC * CP_STEAM * (r['T_out_K'] - T0_REF)
        terms = dict(q_cat_in=q_cat_in, q_feed_in=q_feed_in,
                     q_steam_in=q_steam_in, q_cat_out=q_cat_out,
                     q_oil_out=q_oil_out, q_steam_out=q_steam_out,
                     q_rxn=r['q_rxn'])
        terms['in_total'] = q_cat_in + q_feed_in + q_steam_in
        terms['out_total'] = (q_cat_out + q_oil_out + q_steam_out
                              + r['q_rxn'])
        terms['residual'] = terms['in_total'] - terms['out_total']
        return terms


# =============================================================================
# B.  REGENERATOR (vessel 2)
# =============================================================================
class Regenerator:
    """Single dense-bed regenerator: coke burn from stoichiometry,
    catalyst cooler, wall losses; outlet streams (flue gas, regenerated
    catalyst) leave at the bed temperature T_rgn."""

    T_MIN_K = 850.0      # below this the bed cannot sustain combustion

    @staticmethod
    def solve(CO, y_coke, T_spent_K, q_cool):
        """Closed-form bed temperature from the vessel energy balance.
        All quantities per kg feed; q_cool in kJ/kg feed.
        Returns dict incl. the 'regen_cold' clamp flag."""
        q_comb = y_coke * DH_COKE_V02
        q_loss = Q_LOSS_FRAC * q_comb
        m_air = y_coke * AFR_V02
        m_flue = m_air + y_coke                # all coke leaves as gas
        a_cat = CO * CP_CAT
        a_flue = m_flue * CP_FLUE
        num = (q_comb - q_loss - q_cool
               + a_cat * (T_spent_K - T0_REF)
               + y_coke * CP_COKE * (T_spent_K - T0_REF)
               + m_air * CP_AIR * (T_AIR - T0_REF))
        T_rgn_K = T0_REF + num / (a_cat + a_flue)
        cold = T_rgn_K < Regenerator.T_MIN_K
        if cold:
            T_rgn_K = Regenerator.T_MIN_K
        return dict(T_rgn_K=T_rgn_K, q_comb=q_comb, q_loss=q_loss,
                    q_cool=q_cool, m_air=m_air, m_flue=m_flue,
                    CO=CO, y_coke=y_coke, T_spent_K=T_spent_K,
                    regen_cold=cold)

    @staticmethod
    def energy_balance(g):
        """Enthalpy-stream audit of one regenerator solution (kJ/kg feed).
        Residual ~ 0 unless the bed is clamped at T_MIN_K (regen_cold)."""
        q_cat_in = g['CO'] * CP_CAT * (g['T_spent_K'] - T0_REF)
        q_coke_in = g['y_coke'] * CP_COKE * (g['T_spent_K'] - T0_REF)
        q_air_in = g['m_air'] * CP_AIR * (T_AIR - T0_REF)
        q_cat_out = g['CO'] * CP_CAT * (g['T_rgn_K'] - T0_REF)
        q_flue_out = g['m_flue'] * CP_FLUE * (g['T_rgn_K'] - T0_REF)
        terms = dict(q_cat_in=q_cat_in, q_coke_in=q_coke_in,
                     q_air_in=q_air_in, q_comb=g['q_comb'],
                     q_cat_out=q_cat_out, q_flue_out=q_flue_out,
                     q_cool=g['q_cool'], q_loss=g['q_loss'])
        terms['in_total'] = q_cat_in + q_coke_in + q_air_in + g['q_comb']
        terms['out_total'] = (q_cat_out + q_flue_out
                              + g['q_cool'] + g['q_loss'])
        terms['residual'] = terms['in_total'] - terms['out_total']
        return terms


# =============================================================================
# C.  COUPLED UNIT
# =============================================================================
class RFCCUnit:
    """Riser + regenerator coupled by the catalyst circulation loop.
    The slide-valve controller picks C/O so the riser outlet temperature
    hits its setpoint; the regenerator bed temperature is iterated to
    self-consistency with the resulting coke yield."""

    def __init__(self):
        self.riser = RiserReactor()
        self.regen = Regenerator()

    def _solve_CO(self, T_rgn_K, ROT_set_K, T_pre_K, CCR, co_guess=None):
        """Secant + bisection-fallback C/O solve (same logic as v01)."""
        def gy(CO):
            r = self.riser.solve(CO, T_rgn_K, T_pre_K, CCR)
            return r['T_out_K'] - ROT_set_K, r

        x0 = CO_REF if co_guess is None else min(max(co_guess, CO_MIN),
                                                 CO_MAX)
        x1 = min(x0 + 0.5, CO_MAX) if x0 < CO_MAX else x0 - 0.5
        g0, r0 = gy(x0)
        g1, r1 = gy(x1)
        for _ in range(20):
            if abs(g1) < 0.05:
                return x1, r1, 0
            if g1 == g0:
                break
            x2 = x1 - g1 * (x1 - x0) / (g1 - g0)
            x2 = min(max(x2, CO_MIN), CO_MAX)
            if x2 == x1:
                break
            x0, g0 = x1, g1
            x1 = x2
            g1, r1 = gy(x1)
        g_lo, r_lo = gy(CO_MIN)
        if g_lo >= 0.0:
            return CO_MIN, r_lo, -1
        g_hi, r_hi = gy(CO_MAX)
        if g_hi <= 0.0:
            return CO_MAX, r_hi, +1
        lo, hi = CO_MIN, CO_MAX
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            gm, rm = gy(mid)
            if abs(gm) < 0.05 or hi - lo < 1e-3:
                return mid, rm, 0
            if gm > 0.0:
                hi = mid
            else:
                lo = mid
        return mid, rm, 0

    def solve(self, ROT_C, T_pre_C, Qcool_frac, CCR, feed_frac=1.0,
              T_rgn_init_C=690.0, tol=0.05, max_iter=120):
        """Solve the coupled unit; result dict is v01-compatible and adds
        the per-vessel energy-balance audits."""
        ROT_K = ROT_C + 273.15
        T_pre_K = T_pre_C + 273.15
        F_feed = F_FEED_DESIGN * max(feed_frac, 1e-3)
        q_cool = Qcool_frac * Q_COOL_MAX / F_feed     # kJ/kg feed

        T_rgn = T_rgn_init_C + 273.15
        co_guess = None
        r = g = None
        CO, flag = CO_REF, 0
        for _ in range(max_iter):
            CO, r, flag = self._solve_CO(T_rgn, ROT_K, T_pre_K, CCR,
                                         co_guess)
            co_guess = CO
            g = self.regen.solve(CO, r['yields'][4], r['T_out_K'], q_cool)
            T_new = g['T_rgn_K']
            if abs(T_new - T_rgn) < tol:
                T_rgn = T_new
                break
            T_rgn += 0.5 * (T_new - T_rgn)
        # final consistent pass at the converged bed temperature
        CO, r, flag = self._solve_CO(T_rgn, ROT_K, T_pre_K, CCR, co_guess)
        g = self.regen.solve(CO, r['yields'][4], r['T_out_K'], q_cool)

        y = r['yields']
        T_out_C = r['T_out_K'] - 273.15
        f_dry = float(np.clip(0.15 + 0.004 * (T_out_C - 500.0), 0.08, 0.45))
        F_air = g['m_air'] * F_feed
        F_wet_gas = y[3] * F_feed
        return dict(
            ROT_set=ROT_C, T_pre=T_pre_C, Qcool_frac=Qcool_frac,
            feed_frac=feed_frac, CCR=CCR,
            y_hs=y[0], y_lco=y[1], y_gaso=y[2], y_gas=y[3], y_coke=y[4],
            f_dry=f_dry, conv=1.0 - y[0] - y[1], CO=CO, co_flag=flag,
            T_rgn_C=T_rgn - 273.15, T_out_C=T_out_C,
            T_mix_C=r['T_mix_K'] - 273.15,
            F_feed=F_feed, F_air=F_air, F_wet_gas=F_wet_gas,
            viol_Trgn=max(0.0, (T_rgn - 273.15) - T_RGN_MAX_C),
            viol_air=max(0.0, F_air / AIR_MAX_V02 - 1.0),
            viol_wgc=max(0.0, F_wet_gas / WGC_MAX - 1.0),
            regen_cold=g['regen_cold'],
            eb_reactor=RiserReactor.energy_balance(r),
            eb_regen=Regenerator.energy_balance(g),
        )


_UNIT = RFCCUnit()


def solve_unit(ROT_C, T_pre_C, Qcool_frac, CCR, feed_frac=1.0, **kw):
    """v01-compatible functional interface to the v02 coupled unit."""
    return _UNIT.solve(ROT_C, T_pre_C, Qcool_frac, CCR, feed_frac, **kw)


# =============================================================================
# D.  BASELINE POLICY / SAFETY LAYER (v01 logic on the v02 unit)
# =============================================================================
def _operator_trims(ops, CCR):
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
    while res['co_flag'] == 1 and n < 50:
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
    ops = dict(OPS_DESIGN)
    ops['feed_frac'] = min(1.0, supply)
    ops, res, _ = _operator_trims(ops, CCR)
    return ops, res, economics(res, prices)


def safety_trim(ops, CCR, prices):
    ops, res, _ = _operator_trims(ops, CCR)
    return ops, res, economics(res, prices)


# =============================================================================
# E.  QUICK SELF-TEST
# =============================================================================
if __name__ == '__main__':
    import time
    print(f'coke stoichiometry: CH_{COKE_H_C}, x_CO2={X_CO2}, '
          f'excess air {EXCESS_AIR:.0%}')
    print(f'  -> AFR = {AFR_V02:.2f} kg air/kg coke, '
          f'dH_comb = {DH_COKE_V02/1000:.2f} MJ/kg coke')
    t0 = time.time()
    res = solve_unit(520.0, 220.0, 0.5, CCR=4.0)
    print(f'base case solved in {(time.time()-t0)*1e3:.1f} ms')
    for k in ('conv', 'y_gaso', 'y_coke', 'CO', 'T_rgn_C', 'T_mix_C'):
        print(f'  {k:10s} = {res[k]:.4f}')
    for side in ('eb_reactor', 'eb_regen'):
        eb = res[side]
        print(f'  {side}: in={eb["in_total"]:.1f}  out={eb["out_total"]:.1f}'
              f'  residual={eb["residual"]:+.3f} kJ/kg '
              f'({abs(eb["residual"])/eb["in_total"]*100:.4f} %)')
