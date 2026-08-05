"""
run_final.py - Autonomous final run (Runs 1-5) for the paper revision.

Reuses run_paper.run_strategies() for the canonical per-year,
per-strategy dispatch, and run_robustness.py's variant builders / KPI
helpers for the robustness battery. Each major sub-task is wrapped so a
failure is logged to ../run_log.md (created on first run) and the script
continues with the rest - it does not stop on a sub-task failure.

This is a paper-specific analysis script (not part of the distributable
model) - kept in support/, not model/. It imports the core layers from
../model/.

Run from storageemissions/:
    python support/run_final.py
"""

import os, sys, time, traceback, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model"))

from algorithms import dp
from run import UNIT, GATE_HOUR_CET, LOOKAHEAD_HOURS
from run_paper import (
    load_amer_mix, rolling_dispatch, run_strategies, BASE, PROC,
)
from run_robustness import (
    build_year_bundle, _cheapest_plant, gas_ccgt_names, coal_names,
    variant_biomass_2x, variant_no_mustrun, variant_negative_price, variant_fuel_shock,
    kpi, sign_flip, AMER_NAME,
)

# ============================================================================
# CONSTANTS
# ============================================================================

E_CAP_MWH     = UNIT.e_cap_mwh
PROFIT_COST   = UNIT.cycle_cost_eur_mwh
EMISSION_COST = UNIT.embodied_emission_kg_mwh / 1000.0   # kg/MWh -> kg/kWh
C_EM_CENTRAL  = UNIT.embodied_emission_kg_mwh             # 20.3

YEARS        = list(range(2018, 2026))
CRISIS_YEAR  = 2022
CONTROL_YEAR = 2024

STRATEGIES = ["profit_dp", "emission_dp", "lexico_emissions_dp",
              "lexico_profit_dp", "profit_greedy"]

LOG_PATH = os.path.join(BASE, "../run_log.md")


# ============================================================================
# LOGGING
# ============================================================================

_log_lines = []

def log(msg=""):
    print(msg, flush=True)
    _log_lines.append(msg)

def flush_log():
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")
    _log_lines.clear()

def safe(name, fn, *args, **kwargs):
    """Run fn, catching exceptions. Logs [OK]/[FAILED] and returns
    (True, result) or (False, None). Never raises."""
    try:
        t0 = time.time()
        result = fn(*args, **kwargs)
        log(f"  [OK] {name} ({time.time()-t0:.0f}s)")
        return True, result
    except Exception as e:
        log(f"  [FAILED] {name}: {e}")
        log("```")
        log(traceback.format_exc())
        log("```")
        return False, None


# ============================================================================
# SHARED: custom rolling_dispatch (single-objective, exposes lookahead/n_states)
# ============================================================================

def rolling_dispatch_custom(signal, unit, algo, discharge_cost=0.0,
                             gate_hour=GATE_HOUR_CET, lookahead=LOOKAHEAD_HOURS,
                             n_states=200):
    plan, rows = {}, []
    soc = float(np.clip(0.0, unit.soc_min_mwh, unit.soc_max_mwh))
    for ts in signal.index:
        if ts.hour == gate_hour:
            end_ts = ts + pd.Timedelta(hours=lookahead - 1)
            h_sig = signal.loc[(signal.index >= ts) & (signal.index <= end_ts)]
            if algo is dp:
                acts, cmw, dmw = algo(h_sig.values, unit, soc,
                                       discharge_cost=discharge_cost, n_states=n_states)
            else:
                acts, cmw, dmw = algo(h_sig.values, unit, soc, discharge_cost=discharge_cost)
            for k, t in enumerate(h_sig.index):
                plan[t] = (acts[k], cmw[k], dmw[k])
        act, pc, pd_ = plan.get(ts, (0, 0.0, 0.0))
        c = d = 0.0
        if act == 1 and pc > 0:
            c = min(pc, unit.max_charge_mw(soc)); soc = unit.soc_after_charge(c, soc)
        elif act == -1 and pd_ > 0:
            d = min(pd_, unit.max_discharge_mw(soc)); soc = unit.soc_after_discharge(d, soc)
        soc = float(np.clip(soc, unit.soc_min_mwh, unit.soc_max_mwh))
        act = 1 if c > 1e-9 else (-1 if d > 1e-9 else 0)
        rows.append({"action": act, "charge_mw": c, "discharge_mw": d, "soc_mwh": soc})
    return pd.DataFrame(rows, index=signal.index)


def lexico_p_tie_diagnostics(dam, mei):
    """Re-run Lexico-P's exact rolling dispatch for one year, accumulating
    Bellman-level tie statistics (see algorithms.dp's `tie_stats` param)."""
    tie_stats = {}
    plan = {}
    soc = 0.0
    for ts in dam.index:
        if ts.hour == GATE_HOUR_CET:
            end_ts = ts + pd.Timedelta(hours=LOOKAHEAD_HOURS - 1)
            h_dam = dam.loc[(dam.index >= ts) & (dam.index <= end_ts)]
            h_mei = mei.reindex(h_dam.index).values
            acts, cmw, dmw = dp(h_dam.values, UNIT, soc, discharge_cost=PROFIT_COST,
                                 secondary_signal=h_mei, secondary_discharge_cost=EMISSION_COST,
                                 tie_stats=tie_stats)
            for k, t in enumerate(h_dam.index):
                plan[t] = (acts[k], cmw[k], dmw[k])
        act, pc, pd_ = plan.get(ts, (0, 0.0, 0.0))
        c = d = 0.0
        if act == 1 and pc > 0:
            c = min(pc, UNIT.max_charge_mw(soc)); soc = UNIT.soc_after_charge(c, soc)
        elif act == -1 and pd_ > 0:
            d = min(pd_, UNIT.max_discharge_mw(soc)); soc = UNIT.soc_after_discharge(d, soc)
        soc = float(np.clip(soc, UNIT.soc_min_mwh, UNIT.soc_max_mwh))
    return tie_stats


# ============================================================================
# YEAR BUNDLES
# ============================================================================

def load_all_year_bundles():
    log("\nLoading year bundles 2018-2025 ...")
    amer_mix = load_amer_mix()
    bundles = {}
    for year in YEARS:
        ok, b = safe(f"load bundle {year}", build_year_bundle, year, amer_mix)
        if ok:
            bundles[year] = b
    return bundles


# ============================================================================
# RUN 1 - CANONICAL RESULTS
# ============================================================================

_KPI_RENAME = {
    "total_profit_eur":              "profit_eur",
    "total_charge_mwh":               "charge_mwh",
    "total_discharge_mwh":            "discharge_mwh",
    "net_emissions_kg_co2":           "net_co2_kg",
    "net_emissions_per_mwh_charged":  "co2_per_mwh_charged",
    "n_cycles":                       "equiv_full_cycles",
}

def _remap_kpi_row(rec):
    out = {}
    for k, v in rec.items():
        out[_KPI_RENAME.get(k, k)] = v
    return out


def run1_canonical(year_bundles):
    log("\n" + "=" * 70)
    log("RUN 1 - CANONICAL RESULTS")
    log("=" * 70)

    rows = []
    dispatch_cache = {}   # year -> {strategy: dispatch_df (post compute_emissions)}
    kpi_cache = {}        # (year, strategy) -> unrounded {'profit_eur','net_co2_kg', 'cycles' (discharge-based)}

    for year in YEARS:
        b = year_bundles.get(year)
        if b is None:
            log(f"  {year}: SKIPPED (no data bundle)")
            continue

        def _run_year(b=b, year=year):
            results = run_strategies(b["plants"], b["prices"], b["dam"], b["mei"], b["marginal"], year, tol=1e-6)
            year_disp = {}
            for strat, res in results.items():
                rows.append(_remap_kpi_row(res["kpi"]))
                year_disp[strat] = res["dispatch"]
                td = float(res["dispatch"]["discharge_mw"].sum())
                kpi_cache[(year, strat)] = {
                    "profit_eur": res["kpi"]["total_profit_eur"],
                    "net_co2_kg": res["kpi"]["net_emissions_kg_co2"],
                    "cycles_discharge": td / E_CAP_MWH,
                }
            dispatch_cache[year] = year_disp
            return True

        safe(f"Run 1 - year {year}", _run_year)

    df = pd.DataFrame(rows)
    out_path = os.path.join(PROC, "canonical_kpis.csv")
    df.to_csv(out_path, index=False)
    log(f"  Saved: {out_path}  ({len(df)} rows)")

    # --- Sanity check: profit_dp == lexico_profit_dp (c_em cannot affect them) ---
    log("\n  Sanity check: profit_dp vs lexico_profit_dp (must be identical) ...")
    mismatches = []
    for year in YEARS:
        p = df[(df["year"] == year) & (df["strategy"] == "profit_dp")]
        l = df[(df["year"] == year) & (df["strategy"] == "lexico_profit_dp")]
        if p.empty or l.empty:
            continue
        pr, lr = p.iloc[0], l.iloc[0]
        for col in ["profit_eur", "charge_mwh", "discharge_mwh", "net_co2_kg"]:
            if abs(float(pr[col]) - float(lr[col])) > 1e-6:
                mismatches.append((year, col, pr[col], lr[col]))
    if mismatches:
        log("  *** MISMATCH FOUND - profit_dp and lexico_profit_dp differ! ***")
        for m in mismatches:
            log(f"    year={m[0]} col={m[1]} profit_dp={m[2]} lexico_profit_dp={m[3]}")
    else:
        log("  OK - profit_dp and lexico_profit_dp identical in all years.")

    return df, dispatch_cache, kpi_cache


def run1_fig_exports(dispatch_cache, year_bundles):
    log("\n  Figure-data exports (fig9/10/11, lexico_p_ties) ...")

    strats3 = ["profit_dp", "emission_dp", "lexico_emissions_dp"]

    # --- fig9: dispatch detail, 22-28 Aug 2022 ---
    def _fig9():
        rows = []
        disp22 = dispatch_cache[CRISIS_YEAR]
        dam22, mei22 = year_bundles[CRISIS_YEAR]["dam"], year_bundles[CRISIS_YEAR]["mei"]
        ws, we = "2022-08-22", "2022-08-28 23:00:00"
        for strat in strats3:
            d = disp22[strat].loc[ws:we]
            price = dam22.reindex(d.index)
            mei_w = mei22.reindex(d.index)
            for ts, r in d.iterrows():
                rows.append({
                    "timestamp": ts, "strategy": strat, "soc_mwh": r["soc_mwh"],
                    "charge_mw": r["charge_mw"], "discharge_mw": r["discharge_mw"],
                    "dam_price": price.loc[ts], "mei": mei_w.loc[ts],
                })
        out = pd.DataFrame(rows)
        path = os.path.join(PROC, "fig9_dispatch_week.csv")
        out.to_csv(path, index=False)
        return path
    safe("fig9_dispatch_week.csv", _fig9)

    # --- fig10/11: hourly cumulative profit & net CO2 ---
    def _fig_cum(year, fname):
        disp_y = dispatch_cache[year]
        dam_y = year_bundles[year]["dam"]
        rows = []
        for strat in strats3:
            d = disp_y[strat]
            price = dam_y.reindex(d.index)
            cum_profit = ((d["discharge_mw"] * price) - (d["charge_mw"] * price)).cumsum()
            cum_co2 = d["net_em"].cumsum()
            for ts in d.index:
                rows.append({
                    "timestamp": ts, "strategy": strat,
                    "cum_profit_eur": float(cum_profit.loc[ts]),
                    "cum_net_co2_kg": float(cum_co2.loc[ts]),
                })
        out = pd.DataFrame(rows)
        path = os.path.join(PROC, fname)
        out.to_csv(path, index=False)
        return path
    safe("fig10_cumulative_2022.csv", _fig_cum, CRISIS_YEAR, "fig10_cumulative_2022.csv")
    safe("fig11_cumulative_2025.csv", _fig_cum, 2025, "fig11_cumulative_2025.csv")

    # --- lexico_p_ties_2022.txt ---
    def _ties():
        b22 = year_bundles[CRISIS_YEAR]
        tie_stats = lexico_p_tie_diagnostics(b22["dam"], b22["mei"])
        n_dec = tie_stats.get("n_decisions", 0)
        n_ties = tie_stats.get("n_ties", 0)
        gaps = np.array(tie_stats.get("gaps", []), dtype=float)
        n_nan_gaps = int(np.isnan(gaps).sum())
        tie_rate_pct = 100.0 * n_ties / n_dec if n_dec else float("nan")
        # The raw DAM price series has exactly one NaN hour per year - the EU
        # DST spring-forward transition (last Sunday of March, 02:00 CET does
        # not exist in wall-clock time), present in every one of 2018-2025.
        # NaN comparisons in _is_better() always evaluate False, so this never
        # corrupts the actual dispatch (verified: zero NaN in any dispatch
        # output) - it only poisons a plain np.median() over the raw gap
        # array (200 = 1 hour x n_states NaN entries out of ~2.6M). Use
        # nanmedian and report how many were excluded, for transparency.
        median_gap = float(np.nanmedian(gaps)) if gaps.size else float("nan")
        text = (
            "Lexico-P (profit primary, emission secondary) - Bellman-level tie diagnostics, 2022\n"
            "=================================================================================\n\n"
            "Definition: at every (hour, SOC-grid-point) evaluated in the DP backward pass, the\n"
            "feasible-action primary (profit) values are ranked; a 'tie' is top1-vs-top2 gap <= tol\n"
            "(1e-6 EUR), i.e. a case where the secondary (emission) objective is needed to decide.\n"
            "This counts ALL backward-pass evaluations (the full Bellman recursion), not just the\n"
            "hours actually visited along the realised dispatch trajectory.\n\n"
            f"n_decisions (>=2 feasible actions): {n_dec}\n"
            f"n_ties (gap <= 1e-6 EUR):            {n_ties}\n"
            f"tie_rate_pct:                        {tie_rate_pct:.4f}%\n"
            f"median_value_gap_eur (top1-top2, ALL decisions, not just tied ones): {median_gap:.4f}\n"
            f"  ({n_nan_gaps} of {gaps.size} decisions excluded as NaN - the single EU DST\n"
            "   spring-forward hour each March, where the DAM price series has no value;\n"
            "   confirmed harmless to actual dispatch, see comment in code)\n\n"
            "Interpretation: price is continuous, so exact primary ties are expected to be rare\n"
            "(unlike Lexico-E, where MEI is piecewise-constant - see run_log.md Step 0(b)).\n"
        )
        path = os.path.join(BASE, "../lexico_p_ties_2022.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path
    safe("lexico_p_ties_2022.txt", _ties)


# ============================================================================
# RUN 2 - AMORTISATION SENSITIVITY
# ============================================================================

def run2_benchmark_cem(year_bundles):
    log("\n" + "=" * 70)
    log("RUN 2 - AMORTISATION SENSITIVITY")
    log("=" * 70)

    rows = []
    for year in YEARS:
        b = year_bundles.get(year)
        if b is None:
            continue
        dam, mei = b["dam"], b["mei"]

        def _run_year(dam=dam, mei=mei, year=year):
            for c_em in [0.0, 13.3, 25.4]:
                cost = c_em / 1000.0
                disp_em = rolling_dispatch(mei, UNIT, dp, discharge_cost=cost)
                disp_lexe = rolling_dispatch(mei, UNIT, dp, discharge_cost=cost,
                                              secondary=dam, secondary_discharge_cost=PROFIT_COST)
                for strat, disp in [("emission_dp", disp_em), ("lexico_emissions_dp", disp_lexe)]:
                    k = kpi(disp, dam, mei)
                    rows.append({
                        "year": year, "strategy": strat, "c_em": c_em,
                        "profit_eur": round(k["profit_eur"], 2),
                        "net_co2_kg": round(k["net_co2_kg"], 1),
                        "equiv_full_cycles": round(k["cycles"], 3),
                    })
            return True

        safe(f"Run 2 - year {year}", _run_year)

    df = pd.DataFrame(rows)
    path = os.path.join(PROC, "benchmark_cem.csv")
    df.to_csv(path, index=False)
    log(f"  Saved: {path}  ({len(df)} rows)")
    return df


# ============================================================================
# RUN 3 - NEGATIVE-PRICE EVIDENCE (data query only)
# ============================================================================

def run3_negative_price_evidence(year_bundles):
    log("\n" + "=" * 70)
    log("RUN 3 - NEGATIVE-PRICE EVIDENCE")
    log("=" * 70)
    log("  NOTE: this repository has no hourly actual-generation-by-type or")
    log("  load time series (only installed-capacity snapshots and price data)")
    log(" - see PRE-FLIGHT note in run_log.md. VRE-share-of-load columns (b)")
    log("  are therefore SKIPPED (NaN) for every year; neg_hours and")
    log("  median_neg_price (a, c) are computed from the real DAM price series.")

    rows = []
    for year in YEARS:
        b = year_bundles.get(year)
        if b is None:
            continue

        def _run_year(b=b, year=year):
            dam = b["dam"]
            neg = dam[dam < 0]
            return {
                "year": year,
                "neg_hours": int(len(neg)),
                "frac_vre_gt_60pct_load": float("nan"),
                "frac_vre_gt_40pct_load": float("nan"),
                "median_neg_price": float(neg.median()) if len(neg) else float("nan"),
            }

        ok, row = safe(f"Run 3 - year {year}", _run_year)
        if ok:
            rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(PROC, "negative_price_evidence.csv")
    df.to_csv(path, index=False)
    log(f"  Saved: {path}  ({len(df)} rows)")
    return df


# ============================================================================
# RUN 4 - ROBUSTNESS BATTERY
# ============================================================================

def run4_robustness(year_bundles, dispatch_cache, kpi_cache):
    log("\n" + "=" * 70)
    log("RUN 4 - ROBUSTNESS BATTERY (2022 and 2024, new default config)")
    log("=" * 70)

    rows = []

    def add_row(test, variant, year, strat, k, notes):
        base = kpi_cache.get((year, strat))
        sc = sign_flip(k["net_co2_kg"], base["net_co2_kg"]) if base else None
        rows.append({
            "test": test, "variant": variant, "year": year, "strategy": strat,
            "profit_eur": round(k["profit_eur"], 2),
            "net_co2_kg": round(k["net_co2_kg"], 1),
            "sign_changed": sc,
            "notes": notes,
        })

    # ---- (a) BIOMASS ----
    def _biomass():
        for year in [CRISIS_YEAR, CONTROL_YEAR]:
            b = year_bundles[year]
            _, marg_v, mei_v = variant_biomass_2x(b)
            hours_base = int((b["marginal"] == AMER_NAME).sum())
            hours_var  = int((marg_v == AMER_NAME).sum())
            notes = f"Amer marginal hours base={hours_base} variant={hours_var}"

            disp_p = dispatch_cache[year]["profit_dp"]
            add_row("biomass_2x_fuel_cost", "amer_fuel_cost_2x_coal_proxy", year, "profit_dp",
                    kpi(disp_p, b["dam"], mei_v), notes)

            disp_e_v = rolling_dispatch(mei_v, UNIT, dp, discharge_cost=EMISSION_COST)
            add_row("biomass_2x_fuel_cost", "amer_fuel_cost_2x_coal_proxy", year, "emission_dp",
                    kpi(disp_e_v, b["dam"], mei_v), notes)
        return True
    safe("Run 4a - biomass", _biomass)

    # ---- (b) CHP INFRAMARGINAL BOUND ----
    def _chp():
        for year in [CRISIS_YEAR, CONTROL_YEAR]:
            b = year_bundles[year]
            _, marg_v, mei_v = variant_no_mustrun(b)
            chp_hours_base = int(b["marginal"].isin(b["mustrun_names"]).sum())
            notes = f"base-case hours a must-run CHP unit is marginal={chp_hours_base}; " \
                    f"{len(b['mustrun_names'])} must-run plants removed"

            disp_p = dispatch_cache[year]["profit_dp"]
            add_row("chp_inframarginal_bound", "mustrun_removed_from_pricesetting", year, "profit_dp",
                    kpi(disp_p, b["dam"], mei_v), notes)

            disp_e_v = rolling_dispatch(mei_v, UNIT, dp, discharge_cost=EMISSION_COST)
            add_row("chp_inframarginal_bound", "mustrun_removed_from_pricesetting", year, "emission_dp",
                    kpi(disp_e_v, b["dam"], mei_v), notes)
        return True
    safe("Run 4b - CHP inframarginal bound", _chp)

    # ---- (c) NEGATIVE-PRICE REASSIGNMENT ----
    def _negprice():
        for year in [CRISIS_YEAR, CONTROL_YEAR]:
            b = year_bundles[year]
            rep_gas = _cheapest_plant(b["srmc"], gas_ccgt_names(b))
            _, marg_v, mei_v = variant_negative_price(b, rep_gas)
            n_neg = int((b["dam"] < 0).sum())
            notes = f"{n_neg} negative-price hours reassigned to {rep_gas}"

            disp_p = dispatch_cache[year]["profit_dp"]
            add_row("negative_price_mei", "cheapest_gas_ccgt_mei", year, "profit_dp",
                    kpi(disp_p, b["dam"], mei_v), notes)

            disp_e_v = rolling_dispatch(mei_v, UNIT, dp, discharge_cost=EMISSION_COST)
            add_row("negative_price_mei", "cheapest_gas_ccgt_mei", year, "emission_dp",
                    kpi(disp_e_v, b["dam"], mei_v), notes)
        return True
    safe("Run 4c - negative-price reassignment", _negprice)

    # ---- (d) SCARCITY FORCED IDLE ----
    def _scarcity():
        for year in [CRISIS_YEAR, CONTROL_YEAR]:
            b = year_bundles[year]
            max_srmc = b["srmc"].max(axis=1)
            dam_al = b["dam"].reindex(max_srmc.index)
            scarcity_mask = dam_al > max_srmc
            n_scarcity = int(scarcity_mask.sum())

            disp_v = rolling_dispatch(b["dam"], UNIT, dp, discharge_cost=PROFIT_COST,
                                       idle_mask=scarcity_mask)
            add_row("scarcity_forced_idle", "battery_idle_in_scarcity_hours", year, "profit_dp",
                    kpi(disp_v, b["dam"], b["mei"]),
                    f"{n_scarcity} scarcity hours (price>max modelled SRMC)")
        return True
    safe("Run 4d - scarcity forced idle", _scarcity)

    # ---- (e) FUEL-PRICE FOUR CORNERS ----
    def _fuel_corners():
        for year in [CRISIS_YEAR, CONTROL_YEAR]:
            b = year_bundles[year]
            rep_coal = _cheapest_plant(b["srmc"], coal_names(b))
            rep_gas  = _cheapest_plant(b["srmc"], gas_ccgt_names(b))
            base_cross = int((b["srmc"][rep_coal] < b["srmc"][rep_gas]).sum())
            disp_p = dispatch_cache[year]["profit_dp"]

            for ttf_pct, api2_pct in [(0.10, 0.10), (0.10, -0.10), (-0.10, 0.10), (-0.10, -0.10)]:
                srmc_v, marg_v, mei_v = variant_fuel_shock(b, ttf_pct, api2_pct)
                cross_v = int((srmc_v[rep_coal] < srmc_v[rep_gas]).sum())
                notes = (f"attribution only (dispatch unchanged); coal<gas_ccgt hours "
                         f"base={base_cross} variant={cross_v}")
                add_row("fuel_price_four_corners", f"ttf{ttf_pct:+.0%}_api2{api2_pct:+.0%}",
                        year, "profit_dp", kpi(disp_p, b["dam"], mei_v), notes)
        return True
    safe("Run 4e - fuel-price four corners", _fuel_corners)

    df = pd.DataFrame(rows)
    path = os.path.join(PROC, "robustness_final.csv")
    df.to_csv(path, index=False)
    log(f"  Saved: {path}  ({len(df)} rows)")
    return df


# ============================================================================
# RUN 5 - NUMERICAL CHOICES (2022 only)
# ============================================================================

def run5_numerics(year_bundles, kpi_cache):
    log("\n" + "=" * 70)
    log("RUN 5 - NUMERICAL CHOICES (2022 only)")
    log("=" * 70)

    b22 = year_bundles[CRISIS_YEAR]
    dam, mei = b22["dam"], b22["mei"]
    rows = []

    def add_row(param, value, strat, k):
        rows.append({
            "parameter": param, "value": value, "strategy": strat,
            "profit_eur": round(k["profit_eur"], 2),
            "net_co2_kg": round(k["net_co2_kg"], 1),
            "equiv_full_cycles": round(k["cycles"], 3),
        })

    # baseline (36h, 200 states) reused from Run 1's cached KPIs
    for strat in ["profit_dp", "emission_dp"]:
        base = kpi_cache.get((CRISIS_YEAR, strat))
        if base:
            rows.append({
                "parameter": "rolling_horizon_h", "value": 36, "strategy": strat,
                "profit_eur": round(base["profit_eur"], 2),
                "net_co2_kg": round(base["net_co2_kg"], 1),
                "equiv_full_cycles": round(base["cycles_discharge"], 3),
            })
            rows.append({
                "parameter": "soc_grid_n_states", "value": 200, "strategy": strat,
                "profit_eur": round(base["profit_eur"], 2),
                "net_co2_kg": round(base["net_co2_kg"], 1),
                "equiv_full_cycles": round(base["cycles_discharge"], 3),
            })

    def _horizon():
        for lookahead in [24, 48]:
            disp_p = rolling_dispatch_custom(dam, UNIT, dp, discharge_cost=PROFIT_COST, lookahead=lookahead)
            add_row("rolling_horizon_h", lookahead, "profit_dp", kpi(disp_p, dam, mei))
            disp_e = rolling_dispatch_custom(mei, UNIT, dp, discharge_cost=EMISSION_COST, lookahead=lookahead)
            add_row("rolling_horizon_h", lookahead, "emission_dp", kpi(disp_e, dam, mei))
        return True
    safe("Run 5a - rolling horizon", _horizon)

    def _soc_grid():
        for n_states in [100, 400]:
            disp_p = rolling_dispatch_custom(dam, UNIT, dp, discharge_cost=PROFIT_COST, n_states=n_states)
            add_row("soc_grid_n_states", n_states, "profit_dp", kpi(disp_p, dam, mei))
            disp_e = rolling_dispatch_custom(mei, UNIT, dp, discharge_cost=EMISSION_COST, n_states=n_states)
            add_row("soc_grid_n_states", n_states, "emission_dp", kpi(disp_e, dam, mei))
        return True
    safe("Run 5b - SOC grid resolution", _soc_grid)

    df = pd.DataFrame(rows)
    path = os.path.join(PROC, "numerics_final.csv")
    df.to_csv(path, index=False)
    log(f"  Saved: {path}  ({len(df)} rows)")
    return df


# ============================================================================
# FINAL SUMMARY
# ============================================================================

def write_summary(canonical_df, benchmark_df, negprice_df, robustness_df, numerics_df):
    lines = ["# Run Summary - Autonomous Final Run", ""]

    lines += ["## 1. Step 0 findings", "",
              "See `run_log.md` for the full write-up. Summary: emission_dp's "
              "reward (a) was already correct. Lexico-E's and Lexico-P's secondary "
              "reward (b, c) had a bug - the discharge leg was missing its own cost "
              "term (bare revenue/emission instead of the same Eq. 3.4.1 formula "
              "used by the primary objective). Fixed via a new "
              "`secondary_discharge_cost` parameter on `algorithms.dp()`. "
              "profit_eur reporting (d) was already pure market cashflow, uniform "
              "across all 5 strategies. profit_dp/lexico_profit_dp-primary/greedy "
              "(e) confirmed untouched by c_em.", ""]

    lines += ["## 2. Headline table - Run 1 canonical, 2022 and 2024, all 5 strategies", ""]
    if canonical_df is not None and not canonical_df.empty:
        lines.append("| Year | Strategy | profit_eur | net_co2_kg | equiv_full_cycles |")
        lines.append("|---|---|---:|---:|---:|")
        for year in [CRISIS_YEAR, CONTROL_YEAR]:
            for strat in STRATEGIES:
                sub = canonical_df[(canonical_df["year"] == year) & (canonical_df["strategy"] == strat)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                lines.append(f"| {year} | {strat} | {r['profit_eur']:.2f} | {r['net_co2_kg']:.1f} | "
                             f"{r['equiv_full_cycles']:.1f} |")
    else:
        lines.append("_Run 1 data unavailable._")
    lines.append("")

    lines += ["## 3. Robustness tests - sign verdict (vs Run 1, same year/strategy)", ""]
    if robustness_df is not None and not robustness_df.empty:
        for test in robustness_df["test"].unique():
            sub = robustness_df[robustness_df["test"] == test]
            if sub["sign_changed"].isna().all():
                verdict = "n/a"
            else:
                verdict = "SIGN CHANGES in >=1 case" if sub["sign_changed"].fillna(False).any() else "SURVIVES"
            lines.append(f"- **{test}**: {verdict}")
    else:
        lines.append("_Run 4 data unavailable._")
    lines.append("")

    lines += ["## 4. Run 2 - amortisation sensitivity: retained reduction & cycles removed", ""]
    if benchmark_df is not None and not benchmark_df.empty and canonical_df is not None:
        base20 = canonical_df[canonical_df["strategy"] == "emission_dp"].set_index("year")
        prof = canonical_df[canonical_df["strategy"] == "profit_dp"].set_index("year")
        for c_em in [0.0, 13.3, 25.4]:
            sub = benchmark_df[(benchmark_df["strategy"] == "emission_dp") & (benchmark_df["c_em"] == c_em)]
            # fraction of *emission reduction* (profit_dp net_co2 - emission_dp net_co2) retained
            fracs = []
            for _, r in sub.iterrows():
                yr = r["year"]
                if yr not in prof.index or yr not in base20.index:
                    continue
                p_co2 = prof.loc[yr, "net_co2_kg"]
                e0 = benchmark_df[(benchmark_df["year"] == yr) & (benchmark_df["strategy"] == "emission_dp")
                                  & (benchmark_df["c_em"] == 0.0)]
                if e0.empty:
                    continue
                e0_co2 = e0.iloc[0]["net_co2_kg"]
                ec_co2 = r["net_co2_kg"]
                base_red = p_co2 - e0_co2
                ach_red = p_co2 - ec_co2
                if abs(base_red) > 1e-6:
                    fracs.append(ach_red / base_red)
            if fracs:
                lines.append(f"- c_em={c_em} kg/MWh: mean fraction of c_em=0 reduction retained (full-period avg) "
                             f"= {np.mean(fracs)*100:.1f}%")
        c22 = benchmark_df[(benchmark_df["year"] == CRISIS_YEAR) & (benchmark_df["strategy"] == "emission_dp")]
        if not c22.empty:
            lines.append("")
            lines.append("2022 detail (emission_dp):")
            for _, r in c22.sort_values("c_em").iterrows():
                lines.append(f"  - c_em={r['c_em']}: profit_eur={r['profit_eur']:.2f}, "
                             f"net_co2_kg={r['net_co2_kg']:.1f}, equiv_full_cycles={r['equiv_full_cycles']:.1f}")
    else:
        lines.append("_Run 2 data unavailable._")
    lines.append("")

    def _fmt_cell(v):
        if isinstance(v, float) and float(v).is_integer():
            return str(int(v))
        return str(v)

    lines += ["## 5. Run 3 - negative-price evidence", ""]
    if negprice_df is not None and not negprice_df.empty:
        lines.append("| " + " | ".join(negprice_df.columns) + " |")
        lines.append("|" + "|".join(["---"] * len(negprice_df.columns)) + "|")
        for _, r in negprice_df.iterrows():
            lines.append("| " + " | ".join(_fmt_cell(v) for v in r.values) + " |")
        lines.append("")
        lines.append("_Columns (b) [VRE share of load] are NaN: no hourly actual-generation-by-type "
                     "or load time series exists in this repository (see run_log.md PRE-FLIGHT note "
                     "and Run 3 section) - only installed-capacity snapshots and price data are "
                     "available. neg_hours and median_neg_price are computed from the real DAM series._")
    else:
        lines.append("_Run 3 data unavailable._")
    lines.append("")

    lines += ["## 6. Run 5 - numerical-choice sensitivity (max deviation from base, 2022)", ""]
    if numerics_df is not None and not numerics_df.empty:
        for strat in ["profit_dp", "emission_dp"]:
            base_h = numerics_df[(numerics_df["parameter"] == "rolling_horizon_h") &
                                  (numerics_df["value"] == 36) & (numerics_df["strategy"] == strat)]
            if base_h.empty:
                continue
            base_profit = base_h.iloc[0]["profit_eur"]
            base_co2 = base_h.iloc[0]["net_co2_kg"]
            variants = numerics_df[(numerics_df["strategy"] == strat) &
                                    ~(((numerics_df["parameter"] == "rolling_horizon_h") & (numerics_df["value"] == 36)) |
                                      ((numerics_df["parameter"] == "soc_grid_n_states") & (numerics_df["value"] == 200)))]
            if variants.empty:
                continue
            max_profit_dev = (variants["profit_eur"] - base_profit).abs().max()
            max_co2_dev = (variants["net_co2_kg"] - base_co2).abs().max()
            pct_profit = 100 * max_profit_dev / abs(base_profit) if abs(base_profit) > 1e-6 else float("nan")
            pct_co2 = 100 * max_co2_dev / abs(base_co2) if abs(base_co2) > 1e-6 else float("nan")
            lines.append(f"- **{strat}**: max profit deviation = {pct_profit:.2f}% of base "
                         f"({max_profit_dev:.2f} EUR); max net_co2 deviation = {pct_co2:.2f}% of base "
                         f"({max_co2_dev:.1f} kg)")
    else:
        lines.append("_Run 5 data unavailable._")
    lines.append("")

    lines += ["## 7. Skipped / failed sub-tasks", "",
              "See `run_log.md` for the full [OK]/[FAILED] log of every sub-task. "
              "Known partial completion: Run 3 columns (b) (VRE share of load) are "
              "skipped for all years - no hourly generation-by-type/load data in "
              "this repository (installed-capacity snapshots only).", ""]

    path = os.path.join(BASE, "../run_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\nSaved: {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    t_start = time.time()
    log(f"\n\n{'#'*70}\n# run_final.py execution log\n{'#'*70}")
    flush_log()

    year_bundles = load_all_year_bundles()
    flush_log()

    ok1, res1 = safe("RUN 1 (canonical)", run1_canonical, year_bundles)
    canonical_df, dispatch_cache, kpi_cache = res1 if ok1 else (None, {}, {})
    flush_log()

    if ok1:
        safe("RUN 1 figure exports", run1_fig_exports, dispatch_cache, year_bundles)
        flush_log()

    ok2, benchmark_df = safe("RUN 2 (amortisation sensitivity)", run2_benchmark_cem, year_bundles)
    flush_log()

    ok3, negprice_df = safe("RUN 3 (negative-price evidence)", run3_negative_price_evidence, year_bundles)
    flush_log()

    ok4, robustness_df = safe("RUN 4 (robustness battery)", run4_robustness, year_bundles, dispatch_cache, kpi_cache)
    flush_log()

    ok5, numerics_df = safe("RUN 5 (numerical choices)", run5_numerics, year_bundles, kpi_cache)
    flush_log()

    safe("FINAL summary", write_summary, canonical_df, benchmark_df, negprice_df, robustness_df, numerics_df)
    flush_log()

    log(f"\nTotal runtime: {time.time()-t_start:.0f}s")
    flush_log()
