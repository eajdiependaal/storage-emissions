"""
Layer 3 - Dispatch Algorithms
==============================
The dispatch layer. It takes the MEI signal and merit order from Layer 1
(market.py) and the physical/operational parameters and SOC mechanics of
Layer 2 (storage.py) as given, and decides what action to take each hour - 
the one piece of decision logic in the whole model.

Two algorithms are implemented, both operating on the same StorageUnit
physics (Layer 2) and the same signal series (price or MEI from Layer 1):

  greedy   Evaluates every possible charge-discharge pair in the planning
           horizon and executes the one with the highest reward. Simple
           and transparent, but cannot look ahead past a single cycle.
           Used as a computationally lightweight benchmark.
           Paper: Section 3.4 - Greedy heuristic.

  dp       Solves the Bellman recursion backwards over the full planning
           horizon, then recovers the optimal policy in a forward pass.
           Produces the globally optimal dispatch schedule within the
           horizon. Supports a secondary signal for lexicographic
           tie-breaking (Section 3.4 - Lexicographic objective).
           Paper: Section 3.4 - Dynamic programming.

The paper's dispatch *strategies* (profit-maximising, emission-minimising,
lexicographic, greedy-benchmark) are not separate functions here - they are
constructed by run.py, which chooses which signal (price or MEI) and which
cost (cycle_cost_eur_mwh or embodied_emission_kg_mwh) to pass into `dp()`
or `greedy()` for a given run. This module is agnostic to that choice.

Both functions share the same signature so they can be passed
interchangeably to the rolling_dispatch driver in run.py.

Return format for both:
    actions      list[int]    +1 charge, -1 discharge, 0 idle, per hour
    charge_mw    list[float]  actual charge power [MW] per hour
    discharge_mw list[float]  actual discharge power [MW] per hour

Paper reference: Section 3.4
"""

from __future__ import annotations

import numpy as np

from storage import StorageUnit


# ---------------------------------------------------------------------------
# Greedy  (benchmark)
# ---------------------------------------------------------------------------

def greedy(
    signal:                    np.ndarray,
    unit:                      StorageUnit,
    soc0:                      float,
    discharge_cost:            float = 0.0,
    secondary_signal:          np.ndarray | None = None,   # accepted but not used
    secondary_discharge_cost:  float = 0.0,                # accepted but not used
) -> tuple[list[int], list[float], list[float]]:
    """Find the single best charge-discharge pair in the horizon.

    Algorithm (Section 3.4 - Greedy heuristic):
      1. Charge and discharge power (c_mw, d_mw) depend only on soc0, not
         on which hours are chosen, so compute them once up front (charge
         at soc0, then discharge at whatever SOC results from that charge).
      2. Enumerate all O(n²) (buy hour i, sell hour j) pairs where j > i.
      3. Select the pair maximising the actual reward
         d_mw × (signal[j] − discharge_cost) − c_mw × signal[i],
         not the bare spread signal[j] − signal[i]. c_mw and d_mw
         generally differ (round-trip losses, SOC-dependent capacity), so
         weighting by the bare spread and by the true reward can pick
         different pairs.
      4. Execute that pair only if its reward exceeds 0 (better than
         idling); otherwise do nothing.

    This gives the globally optimal single-cycle dispatch for the given
    soc0. It cannot capture multi-cycle opportunities - that is the DP's
    advantage.

    Parameters
    ----------
    signal : np.ndarray
        Hourly primary signal - price [EUR/MWh] or MEI [kg/kWh_e].
    unit : StorageUnit
        Storage parameters (Layer 2).
    soc0 : float
        State of charge at the start of the horizon [MWh].
    discharge_cost : float
        Cost subtracted from the discharge-leg reward, in the same units
        as `signal` (Section 3.4, Eq. 3.4.1) - e.g. unit.cycle_cost_eur_mwh
        for a price signal. Default 0 (no cost).
    secondary_signal, secondary_discharge_cost : optional
        Ignored for greedy; present so greedy and dp share a signature.

    Returns
    -------
    actions, charge_mw, discharge_mw : lists of length n.
    """
    n      = len(signal)
    actions      = [0]   * n
    charge_mw    = [0.0] * n
    discharge_mw = [0.0] * n

    # --- Charge/discharge power depend only on soc0 (fixed for this call) ---
    c_mw = unit.max_charge_mw(soc0)
    soc_after = unit.soc_after_charge(c_mw, soc0)
    d_mw = unit.max_discharge_mw(soc_after)

    # --- Find best (charge hour i, discharge hour j) pair ---
    best_reward = 0.0   # must beat idling (reward 0) to be worth executing
    best_i: int | None = None
    best_j: int | None = None

    for i in range(n - 1):
        for j in range(i + 1, n):
            reward = d_mw * (signal[j] - discharge_cost) - c_mw * signal[i]
            if reward > best_reward:
                best_reward = reward
                best_i, best_j = i, j

    # --- Execute the pair (if one was found) ---
    if best_i is not None:
        actions[best_i]      = 1
        actions[best_j]      = -1
        charge_mw[best_i]    = c_mw
        discharge_mw[best_j] = d_mw

    return actions, charge_mw, discharge_mw


# ---------------------------------------------------------------------------
# Dynamic programming  (main algorithm)
# ---------------------------------------------------------------------------

def dp(
    signal:                    np.ndarray,
    unit:                      StorageUnit,
    soc0:                      float,
    discharge_cost:            float = 0.0,
    secondary_signal:          np.ndarray | None = None,
    secondary_discharge_cost:  float = 0.0,
    n_states:                  int = 200,
    tol:                       float = 1e-6,
    force_idle:                np.ndarray | None = None,
    tie_stats:                 dict | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Optimal dispatch via backward dynamic programming.

    Solves the Bellman recursion (Section 3.4, Eq. 3.4):
        V_t(s) = max_{a ∈ A(s)} [ r_t(s, a) + V_{t+1}(s') ]

    Terminal condition:
        V_T(s) = 0  for all SOC states s.
    (No terminal reward assumed - finite horizon, no end-of-period value.)

    Immediate rewards (Eq. 3.4.1 / 3.4.2), where c_cycle = `discharge_cost`:
        Discharge: r_t = P_d × (λ_t − c_cycle)
        Charge:    r_t = −P_c × λ_t
        Idle:      r_t = 0

    `signal` and `discharge_cost` must be in the same units - e.g. pass
    price [EUR/MWh] with unit.cycle_cost_eur_mwh for the profit objective,
    or MEI [kg CO2/kWh_e] with unit.embodied_emission_kg_mwh / 1000 for
    the emission objective (kg/MWh → kg/kWh). The DP itself is agnostic
    to which objective `signal` represents.

    For the lexicographic objective (Section 3.4 - Lexicographic):
    provide `secondary_signal` (the other of price/MEI) and
    `secondary_discharge_cost` (that signal's own discharge-leg cost - 
    e.g. pass unit.cycle_cost_eur_mwh as the secondary cost when price is
    the secondary signal). The secondary reward is computed by the exact
    same Eq. 3.4.1/3.4.2 formula as the primary, just with signal/cost
    swapped - it is never a bare revenue/emission term. The primary
    objective is optimised first; the secondary is used only to break
    ties within a numerical tolerance of 1 × 10⁻⁶. Note that MEI is
    piecewise-constant (one value per marginal plant), so exact ties are
    common when MEI is the *primary* signal - unlike price, which is
    continuous and ties only by coincidence.

    Implementation notes
    --------------------
    - The SOC axis is discretised into `n_states` evenly spaced levels.
    - SOC transitions from charge/discharge land between grid points;
      the future value is recovered by linear interpolation (np.interp).
    - The forward pass snaps to the nearest grid state at each hour to
      read the stored policy, then applies exact SOC physics.

    Parameters
    ----------
    signal : np.ndarray
        Primary hourly signal - price [EUR/MWh] or MEI [kg/kWh_e].
    unit : StorageUnit
        Storage parameters (Layer 2).
    soc0 : float
        State of charge at the start of the horizon [MWh].
    discharge_cost : float
        Cost subtracted from the discharge-leg reward (c_cycle in
        Eq. 3.4.1), in the same units as `signal`. Default 0 (no cost).
    secondary_signal : np.ndarray, optional
        Secondary signal for lexicographic tie-breaking.
        When None, only the primary objective is used.
    secondary_discharge_cost : float
        Cost subtracted from the secondary reward's discharge leg, in the
        same units as `secondary_signal`. Only meaningful when
        `secondary_signal` is provided. Default 0.
    n_states : int
        Number of SOC grid points (default 200).
        More states → higher accuracy but longer runtime.
        200 is sufficient for a 1 MWh unit with 1 MW power.
    force_idle : np.ndarray[bool], optional
        Per-hour mask, same length as `signal`. Hours where True are
        constrained to the idle action (no charge or discharge), e.g. to
        model scarcity-price hours in which the unit is held offline.
        When None, all hours are freely optimised.
    tie_stats : dict, optional
        When provided (an empty dict), populated in place with tie
        diagnostics accumulated over every (hour, SOC-grid-point)
        evaluated in the backward pass: 'n_decisions' (feasible-action
        comparisons with >=2 candidates), 'n_ties' (of those, how many
        had a primary-value gap <= tol, i.e. needed the secondary to
        decide) and 'gaps' (the list of top1-vs-top2 primary-value gaps,
        one per decision). Only meaningful when `secondary_signal` is
        provided; adds bookkeeping overhead, so leave as None for normal
        dispatch runs.

    Returns
    -------
    actions, charge_mw, discharge_mw : lists of length n.
    """
    n        = len(signal)
    c_cycle  = discharge_cost
    c_cycle2 = secondary_discharge_cost
    use_lex  = secondary_signal is not None
    idle_mask = force_idle if force_idle is not None else np.zeros(n, dtype=bool)

    # Discrete SOC grid spanning [SOC_min, SOC_max].
    soc_grid = np.linspace(unit.soc_min_mwh, unit.soc_max_mwh, n_states)

    # Value function tables.
    # V_p[t, s] = best PRIMARY objective from hour t to end, at grid state s.
    # V_s[t, s] = best SECONDARY objective for the same trajectory.
    # Both are zero at the terminal time step (no end-of-period value).
    V_p = np.zeros((n + 1, n_states))
    V_s = np.zeros((n + 1, n_states))

    # policy[t, s] stores the best action index: +1 charge, -1 discharge, 0 idle.
    policy = np.zeros((n, n_states), dtype=np.int8)

    # ---------------------------------------------------------------
    # Backward pass - fill V_p and V_s from t = T-1 down to t = 0.
    # ---------------------------------------------------------------
    for t in range(n - 1, -1, -1):
        sig  = signal[t]
        sig2 = secondary_signal[t] if use_lex else 0.0
        t_idle = bool(idle_mask[t])

        for s, soc in enumerate(soc_grid):

            # Initialise with the idle action (r = 0, SOC unchanged).
            best_vp = 0.0 + np.interp(soc, soc_grid, V_p[t + 1])
            best_vs = 0.0 + np.interp(soc, soc_grid, V_s[t + 1])
            best_a  = 0
            candidate_vps = [best_vp]   # feasible-action primary values, for tie diagnostics

            if t_idle:
                # Forced idle hour (e.g. scarcity-price masking): skip the
                # charge/discharge branches, only the idle action is legal.
                V_p[t, s]    = best_vp
                V_s[t, s]    = best_vs
                policy[t, s] = best_a
                continue

            # --- Try: charge ---
            # Eq. 3.4.2 - no cost term on the charge leg, for either objective.
            p_c = unit.max_charge_mw(soc)
            if p_c > 1e-9:
                soc_next = unit.soc_after_charge(p_c, soc)   # Eq. 3.3.1
                r_p = -p_c * sig
                r_s = -p_c * sig2
                vp  = r_p + np.interp(soc_next, soc_grid, V_p[t + 1])
                vs  = r_s + np.interp(soc_next, soc_grid, V_s[t + 1])
                candidate_vps.append(vp)
                if _is_better(vp, vs, best_vp, best_vs, use_lex, tol):
                    best_vp, best_vs, best_a = vp, vs, 1

            # --- Try: discharge ---
            # Eq. 3.4.1 - cost applied to whichever objective's reward this
            # is: c_cycle for the primary, c_cycle2 for the secondary. The
            # secondary is the *same* reward formula as the primary, not a
            # bare revenue/emission term - e.g. for Lexico-E (primary=MEI,
            # secondary=price), the secondary must include the profit
            # cycle cost exactly as profit_dp's own primary reward does.
            p_d = unit.max_discharge_mw(soc)
            if p_d > 1e-9:
                soc_next = unit.soc_after_discharge(p_d, soc)  # Eq. 3.3.2
                r_p = p_d * (sig - c_cycle)
                r_s = p_d * (sig2 - c_cycle2)
                vp  = r_p + np.interp(soc_next, soc_grid, V_p[t + 1])
                vs  = r_s + np.interp(soc_next, soc_grid, V_s[t + 1])
                candidate_vps.append(vp)
                if _is_better(vp, vs, best_vp, best_vs, use_lex, tol):
                    best_vp, best_vs, best_a = vp, vs, -1

            if tie_stats is not None and use_lex and len(candidate_vps) >= 2:
                top1, top2 = sorted(candidate_vps, reverse=True)[:2]
                gap = top1 - top2
                tie_stats["n_decisions"] = tie_stats.get("n_decisions", 0) + 1
                tie_stats.setdefault("gaps", []).append(gap)
                if gap <= tol:
                    tie_stats["n_ties"] = tie_stats.get("n_ties", 0) + 1

            V_p[t, s]    = best_vp
            V_s[t, s]    = best_vs
            policy[t, s] = best_a

    # ---------------------------------------------------------------
    # Forward pass - recover the optimal policy from the initial SOC.
    # ---------------------------------------------------------------
    actions:      list[int]   = []
    charge_mw:    list[float] = []
    discharge_mw: list[float] = []

    soc = float(np.clip(soc0, unit.soc_min_mwh, unit.soc_max_mwh))

    for t in range(n):
        # Snap current SOC to the nearest grid state to look up the policy.
        s = int(np.argmin(np.abs(soc_grid - soc)))
        a = int(policy[t, s])

        c_mw = 0.0
        d_mw = 0.0

        if a == 1:                               # charge
            c_mw = unit.max_charge_mw(soc)
            soc  = unit.soc_after_charge(c_mw, soc)
        elif a == -1:                            # discharge
            d_mw = unit.max_discharge_mw(soc)
            soc  = unit.soc_after_discharge(d_mw, soc)

        # Numerical guard: keep SOC within physical bounds.
        soc = float(np.clip(soc, unit.soc_min_mwh, unit.soc_max_mwh))

        actions.append(a)
        charge_mw.append(c_mw)
        discharge_mw.append(d_mw)

    return actions, charge_mw, discharge_mw


# ---------------------------------------------------------------------------
# Helper: lexicographic comparison
# ---------------------------------------------------------------------------

def _is_better(
    vp: float, vs: float,
    best_vp: float, best_vs: float,
    use_secondary: bool,
    tol: float = 1e-6,
) -> bool:
    """Return True if (vp, vs) is lexicographically better than (best_vp, best_vs).

    Primary objective vp always takes precedence.
    The secondary objective vs is used only to break ties within `tol`.
    When use_secondary is False, ties in vp are broken in favour of the
    current best (i.e., the existing action is kept).

    Paper reference: Section 3.4 - Lexicographic objective.
    """
    if vp > best_vp + tol:
        return True                              # strictly better primary
    if use_secondary and abs(vp - best_vp) <= tol and vs > best_vs:
        return True                              # tied primary, better secondary
    return False
