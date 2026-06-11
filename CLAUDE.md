# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scientific computing research project simulating **LDPE (Low-Density Polyethylene) free-radical polymerisation** in industrial tubular (PFR) and autoclave (CSTR) reactors. The pipeline progresses through four stages: reaction kinetics → dynamic PFR simulation with temperature effects → reinforcement learning optimisation → CSTR modelling.

The authoritative technical reference is `REPORT_LDPE_PFR_Simulation.md`, which documents the physics, debugging history, and validated parameter values.

## Running the Code

No build or install step is required. Dependencies are NumPy, SciPy, and Matplotlib (all standard scientific Python).

```bash
# Run baseline PFR simulation (~1–2 s, produces 8 PNG figures)
python step2_rxn_w_T_effect_v05.py

# Train RL agent (~5–10 min, produces training-curve and trajectory PNGs)
python step3_rl_optimization_v06.py

# Run CSTR/autoclave model
python step4_cstr_polymerization.py
```

All scripts set `matplotlib.use('Agg')` at the top and save figures as PNGs — no display server required.

## Architecture

The codebase follows a linear development history (v01 → v06). **Always work from the highest-numbered version of each step** unless specifically debugging an earlier iteration.

### Core files (current versions)

| File | Purpose |
|---|---|
| `step2_rxn_w_T_effect_v05.py` | High-fidelity PFR simulator; exposes `run_pfr(...)` |
| `step3_rl_optimization_v06.py` | REINFORCE agent over 5D operating-condition space |
| `step4_cstr_polymerization.py` | CSTR (autoclave) model, 11 scalar ODEs |
| `step1_rxn_kinetics_v02.py` | Standalone kinetics reference (superseded by step2) |

### Physical model

Each grid node (or the single CSTR cell) tracks **11 state variables**:
- λ₀, λ₁, λ₂ — moments of living radicals
- μ₀, μ₁, μ₂ — moments of dead polymer
- [I], [M], T, Tc, [CTA] — initiator, monomer, reactor temp, coolant temp, chain-transfer agent

Seven reaction classes use Arrhenius kinetics: initiation, propagation, termination (combination + disproportionation), transfer to monomer, transfer to polymer (long-chain branching), transfer to CTA, and β-scission.

### PFR numerical scheme

- **Solver:** `scipy.integrate.solve_ivp(..., method='Radau')` — implicit stiff solver
- **Spatial discretisation:** Backward upwind finite differences, N=200 nodes over L=500 m
- **Sparse Jacobian:** Pre-computed sparsity pattern for the N×11 state vector (essential for performance)
- **Moment closure:** μ₃ uses `max(Hulburt–Katz, conservative lower bound)` — both branches must stay intact to avoid solver failure in high-conversion regimes

### RL agent (`step3`)

- **Algorithm:** REINFORCE (policy gradient) with moving-average baseline and Adam optimiser — pure NumPy, no RL library
- **Policy network:** 2-layer MLP, tanh activations
- **State (9D):** normalised [X, T_peak, Mn, PDI, T₀, log(ini₀), Tc, U, cta₀]
- **Action (5D):** tanh-squashed deltas over [T₀, log(ini₀), Tc, U, cta₀]
- **Fast simulator:** Uses `run_pfr` with N=20, rtol=5e-3 during training for speed (introduces intentional numerical noise)
- **Reward weights:** X:15, T_peak:12, Mn:10, Mw:6, PDI:5 — retuning needed if kinetics change

## Critical Constraints

### Kinetics synchronisation
Arrhenius parameters (`A_k*`, `Ea_k*`) appear independently in `step2_v05`, `step3_v06`, and `step4`. Any change to rate constants **must be applied to all three files** or the RL agent will train against a different physics than the validator uses.

### Validated baseline parameters
These values were carefully calibrated against industrial literature (see `REPORT_LDPE_PFR_Simulation.md`). Do not change without understanding the consequences:

| Parameter | Value | Reason |
|---|---|---|
| T₀ | 195 °C | k_d half-life ≈ τ_res = 41.7 s |
| ini₀ | 0.01 mol/m³ | Industrial range; higher values cause thermal runaway |
| A_ktrm | 1550 | Effective CTA proxy; Cs = 2.63×10⁻⁴ (lit: 10⁻⁴–3×10⁻³) |
| U_heat | 1500 W/(m²K) | Matched to literature T_peak |
| Tc_in | 130 °C | Paired with U_heat for X ≈ 5–15% |

### Grid resolution
N < 50 on a 500 m reactor masks sub-grid hot-spots (numerical diffusion). Use **N ≥ 100 for validation**; N=20 is only appropriate for RL training speed.

### CTA toggle
Setting `cta_0 = 0` disables the CTA mechanism cleanly — all `ktr_cta * cta * (...)` terms collapse to zero. This is the correct way to compare runs with and without CTA.

## Validation Targets

From `REPORT_LDPE_PFR_Simulation.md` — results to match after any significant change:

| Metric | Validated result | Industrial target |
|---|---|---|
| Monomer conversion X | 5.75% | 5–15% |
| Peak temperature T_peak | 299.1 °C | 200–300 °C |
| Mn | 63.8 kg/mol | 30–150 kg/mol |
| Mw | 134.4 kg/mol | 100–500 kg/mol |
| PDI | 2.11 | 3–15 (model limitation: single zone) |

Validation is by **visual inspection of PNG outputs** — there is no automated test suite.
