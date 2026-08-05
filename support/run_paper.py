"""
run_paper.py - Full paper results pipeline
==========================================
Produces all outputs for the paper:
  Step 1  Annual runs 2018-2025
  Step 3  Summary tables
  Step 4  Sensitivity analysis (CO2 price + battery size, 2022)
  Step 5  All paper figures
  Step 6  Gas crisis analysis console table
  Step 8  RESULTS_SUMMARY.md

This is a paper-specific analysis script (not part of the distributable
model) - kept in support/, not model/. It imports the core layers from
../model/.

Run from storageemissions/:
    python support/run_paper.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# model imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model"))
from market import (load_plants, build_mei_with_blends,
                    apply_cofiring_blend as _apply_blend, normalize_fuel)
from storage import StorageUnit
from algorithms import greedy, dp

BASE        = os.path.dirname(os.path.abspath(__file__))
BZNL        = os.path.join(BASE, "../data/raw/BZ_NL")
PROC        = os.path.join(BASE, "../data/processed/model")
FIG_DIR     = os.path.join(PROC, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PROC, exist_ok=True)

# battery config and shared settings: imported from run.py
# run.py is the single source of truth for all model parameters. Any change
# there propagates here automatically.
from run import (UNIT, GATE_HOUR_CET, LOOKAHEAD_HOURS, FUEL_PRICE_MAP)

GATE_HOUR  = GATE_HOUR_CET
LOOKAHEAD  = LOOKAHEAD_HOURS
E_CAP_MWH  = UNIT.e_cap_mwh              # alias for n_cycles and plot scaling
_RTE       = UNIT.eta_c * UNIT.eta_d     # η_RT - used in sensitivity runs

# colour scheme
FUEL_COLOR = {
    # Coal
    "coal":              "#6B3A2A",
    # Biomass (lime-green: distinct from coal brown and RES forest-green)
    "biomass":           "#A8C66C",
    # Gas CCGT (utility / merchant)
    "gas":               "#2166AC",
    "gas_ccgt":          "#2166AC",
    # Gas CCGT CHP (industrial / district heating)
    "gas_ccgt_chp":      "#74ADD1",
    # Gas OCGT (peakers)
    "gas_ocgt":          "#41B6C4",
    # Gas Engine CHP (reciprocating engine, e.g. NAM)
    "gas_engine_chp":    "#A6CEE3",
    # Blast furnace gas (Velsen)
    "blast_furnace_gas": "#762A83",
    # Nuclear
    "nuclear":           "#F6B50A",
    # Renewables
    "wind":              "#4DAF4A",
    "solar":             "#4DAF4A",
    "res":               "#4DAF4A",
    "other":             "#AAAAAA",
}

# Map from ORIGINAL CSV fuel strings → display key (before normalize_fuel is called)
_DISPLAY_MAP = {
    "Coal (supercritical)":       "coal",
    "Coal (ultra-supercritical)": "coal",
    "Coal (subcritical)":         "coal",
    "Biomass (co-firing)":        "biomass",
    "Gas CCGT":                   "gas_ccgt",
    "Gas CCGT CHP":               "gas_ccgt_chp",
    "Gas OCGT":                   "gas_ocgt",
    "Gas Engine CHP":             "gas_engine_chp",
    "Blast furnace gas":          "blast_furnace_gas",
    "Nuclear":                    "nuclear",
    "Solar + Wind":               "res",
}

# Map from CANONICAL (post-normalize_fuel) keys → display key (fallback)
_CANONICAL_COLOR = {
    "gas":               "gas_ccgt",         # default: CCGT blue
    "coal":              "coal",
    "biomass":           "biomass",
    "blast_furnace_gas": "blast_furnace_gas",
    "nuclear":           "nuclear",
    "wind":              "res",
    "solar":             "res",
}


def display_fuel(fuel_str: str) -> str:
    """Return a display-level fuel key.
    Accepts both original CSV labels ('Gas CCGT CHP') and
    post-normalize_fuel canonical keys ('gas').
    """
    s = str(fuel_str).strip()
    # 1. Try original CSV label
    if s in _DISPLAY_MAP:
        return _DISPLAY_MAP[s]
    # 2. Try canonical key directly
    s_lower = s.lower().replace("-", " ").replace("_", " ")
    for canon_key, display_key in _CANONICAL_COLOR.items():
        if s_lower == canon_key.replace("_", " "):
            return display_key
    # 3. Normalize then look up
    from market import normalize_fuel
    canon = normalize_fuel(s)
    return _CANONICAL_COLOR.get(canon, "other")


def load_raw_fuels(year: int) -> dict:
    """Return {plant_name: original_fuel_string} from the yearly CSV,
    BEFORE load_plants() normalises the fuel column.
    Used for display/colour purposes only.
    """
    yfile = os.path.join(BZNL, f"meritorder_NL_{year}.csv")
    base  = os.path.join(BZNL, "meritorder_NL.csv")
    path  = yfile if os.path.exists(yfile) else base
    df    = pd.read_csv(path)
    return df.set_index("name")["fuel"].to_dict()

STRAT_COLOR = {
    "profit_dp":           "#D32F2F",   # red
    "emission_dp":         "#2E7D32",   # dark green
    "lexico_emissions_dp": "#1565C0",   # dark blue  (distinct from green)
    "lexico_profit_dp":    "#E65100",   # deep orange (distinct from red)
    "profit_greedy":       "#757575",   # grey
}
STRAT_LABEL = {
    "profit_dp":           "Profit max (DP)",
    "emission_dp":         "Emission min (DP)",
    "lexico_emissions_dp": "Lexico-E: emission primary",
    "lexico_profit_dp":    "Lexico-P: profit primary",
    "profit_greedy":       "Greedy benchmark",
}


plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.3, "font.family": "sans-serif",
})

# ============================================================================
# DATA LOADING
# ============================================================================

def plant_fuel_canonical(fuel_str: str) -> str:
    return normalize_fuel(fuel_str)


def load_prices():
    raw = pd.read_csv(
        os.path.join(BZNL, "commodity_prices_BZNL.csv"),
        parse_dates=["Datum"], index_col="Datum",
    )
    raw = raw.rename(columns={"Kolen": "Coal",
                               "blast furnace gas": "Blast_furnace_gas"})
    idx = pd.date_range(raw.index.min(),
                        raw.index.max() + pd.Timedelta(hours=23), freq="h")
    return raw.reindex(idx).ffill()


def load_dam():
    raw = pd.read_csv(os.path.join(BZNL, "BZ_NL.csv"))
    raw["datetime"] = (
        raw["Date"].str.split(" - ").str[0]
        .str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
    )
    raw["datetime"] = pd.to_datetime(raw["datetime"], format="%d/%m/%Y %H:%M:%S")
    raw = raw.set_index("datetime").sort_index()
    return raw["Price"].astype(float).resample("h").mean()


def load_amer_mix():
    return pd.read_csv(
        os.path.join(BZNL, "amer_fuel_mix.csv"),
        comment="#", parse_dates=["date_from"],
    ).sort_values("date_from").reset_index(drop=True)


def load_data_for_year(year: int, prices_override: pd.DataFrame = None):
    """Return (plants, prices, dam_price) for a specific year, using
    the year-specific merit order file and the daily commodity prices."""
    # Plant stack
    yfile = os.path.join(BZNL, f"meritorder_NL_{year}.csv")
    base_file = os.path.join(BZNL, "meritorder_NL.csv")
    stack_path = yfile if os.path.exists(yfile) else base_file
    plants = load_plants(stack_path)

    prices = prices_override if prices_override is not None else load_prices()
    dam    = load_dam()

    # Filter to year
    start, end = f"{year}-01-01", f"{year}-12-31"
    prices = prices.loc[start:end]
    dam    = dam.loc[start:end]
    idx    = prices.index.intersection(dam.index)
    return plants, prices.loc[idx], dam.loc[idx]


# FUEL_PRICE_MAP imported from run.py above

def apply_cofiring_blend(plants, period_start, amer_mix):
    """Thin wrapper: apply Amer blend via market.apply_cofiring_blend."""
    return _apply_blend(plants, "Amer (Amercentrale)", period_start, amer_mix)


def build_mei_for_period(plants, prices, dam_price, amer_mix):
    """Run the full MEI pipeline with Amer cofiring splits via market.build_mei_with_blends."""
    mixes = {"Amer (Amercentrale)": amer_mix} if amer_mix is not None else {}
    return build_mei_with_blends(plants, prices, dam_price, FUEL_PRICE_MAP, mixes)


# ============================================================================
# DISPATCH HELPERS
# ============================================================================

def rolling_dispatch(signal, unit, algo, discharge_cost=0.0, secondary=None,
                      secondary_discharge_cost=0.0, tol=1e-6, idle_mask=None, tie_stats=None):
    plan, rows = {}, []
    soc = float(np.clip(0.0, unit.soc_min_mwh, unit.soc_max_mwh))
    for ts in signal.index:
        if ts.hour == GATE_HOUR:
            end_ts = ts + pd.Timedelta(hours=LOOKAHEAD - 1)
            h_sig  = signal.loc[(signal.index >= ts) & (signal.index <= end_ts)]
            h_sec  = secondary.reindex(h_sig.index).values if secondary is not None else None
            if algo is dp:
                h_idle = (idle_mask.reindex(h_sig.index).fillna(False).values
                          if idle_mask is not None else None)
                acts, cmw, dmw = algo(h_sig.values, unit, soc, discharge_cost=discharge_cost,
                                      secondary_signal=h_sec, secondary_discharge_cost=secondary_discharge_cost,
                                      tol=tol, force_idle=h_idle, tie_stats=tie_stats)
            else:
                acts, cmw, dmw = algo(h_sig.values, unit, soc, discharge_cost=discharge_cost,
                                      secondary_signal=h_sec, secondary_discharge_cost=secondary_discharge_cost)
            for k, t in enumerate(h_sig.index):
                plan[t] = (acts[k], cmw[k], dmw[k])
        act, pc, pd_ = plan.get(ts, (0, 0.0, 0.0))
        c = d = 0.0
        if act == 1 and pc > 0:
            c   = min(pc, unit.max_charge_mw(soc))
            soc = unit.soc_after_charge(c, soc)
        elif act == -1 and pd_ > 0:
            d   = min(pd_, unit.max_discharge_mw(soc))
            soc = unit.soc_after_discharge(d, soc)
        soc = float(np.clip(soc, unit.soc_min_mwh, unit.soc_max_mwh))
        act = 1 if c > 1e-9 else (-1 if d > 1e-9 else 0)
        rows.append({"action": act, "charge_mw": c, "discharge_mw": d, "soc_mwh": soc})
    return pd.DataFrame(rows, index=signal.index)


def compute_emissions(dispatch, mei):
    o = dispatch.copy()
    o["mei"]              = mei.reindex(o.index)
    o["charge_em"]        = o["charge_mw"]    * o["mei"] * 1000.0
    o["discharge_em"]     = o["discharge_mw"] * o["mei"] * 1000.0
    o["net_em"]           = o["charge_em"] - o["discharge_em"]
    o["net_em_cum_tco2"]  = o["net_em"].cumsum() / 1000.0
    return o


def fuel_of_plant(name, plants, raw_fuels=None):
    """Return display-level fuel key for a plant (used for colours and tracking).
    raw_fuels: optional {name: original_csv_fuel_string} for granular display.
    Falls back to the normalized fuel in plants (which loses CCGT/CHP distinction).
    """
    if raw_fuels and name in raw_fuels:
        return display_fuel(raw_fuels[name])
    if name not in plants.index:
        return "other"
    # plants.fuel is already normalized by load_plants() → use canonical fallback
    return display_fuel(str(plants.loc[name, "fuel"]))


TRACK_FUELS = [
    # (tracking_name, set_of_display_keys_that_count)
    ("coal",              {"coal"}),
    ("biomass",           {"biomass"}),
    ("gas_ccgt",          {"gas_ccgt", "gas"}),
    ("gas_ccgt_chp",      {"gas_ccgt_chp"}),
    ("gas_ocgt",          {"gas_ocgt"}),
    ("blast_furnace_gas", {"blast_furnace_gas"}),
    ("nuclear",           {"nuclear"}),
    ("res",               {"wind", "solar", "res"}),
]

def marginal_fuel_hours(dispatch, marginal, plants):
    """Return dict of fuel_type -> hours charged / discharged."""
    out = {}
    for action, label in [(1, "charged"), (-1, "discharged")]:
        sub   = dispatch[dispatch["action"] == action]
        fuels = marginal.reindex(sub.index).apply(
            lambda n: fuel_of_plant(str(n), plants)
        )
        for track_name, key_set in TRACK_FUELS:
            col      = f"hours_{label}_{track_name}_marginal"
            out[col] = int(fuels.isin(key_set).sum())
    return out


def run_strategies(plants, prices, dam_price, mei, marginal, year, tol=1e-6):
    profit_cost   = UNIT.cycle_cost_eur_mwh
    emission_cost = UNIT.embodied_emission_kg_mwh / 1000.0   # kg/MWh -> kg/kWh, matches MEI scale

    strategies = {
        "profit_dp":           rolling_dispatch(dam_price, UNIT, dp, discharge_cost=profit_cost, tol=tol),
        "emission_dp":         rolling_dispatch(mei,       UNIT, dp, discharge_cost=emission_cost, tol=tol),
        # Secondary reward uses the SAME Eq. 3.4.1/3.4.2 formula as that
        # objective's own primary reward - i.e. Lexico-E's price secondary
        # includes the profit cycle cost, and Lexico-P's emission secondary
        # includes the embodied cost. Not a bare revenue/emission term.
        "lexico_emissions_dp": rolling_dispatch(mei,       UNIT, dp, discharge_cost=emission_cost,
                                                secondary=dam_price, secondary_discharge_cost=profit_cost, tol=tol),
        "lexico_profit_dp":    rolling_dispatch(dam_price, UNIT, dp, discharge_cost=profit_cost,
                                                secondary=mei, secondary_discharge_cost=emission_cost, tol=tol),
        "profit_greedy":       rolling_dispatch(dam_price, UNIT, greedy, discharge_cost=profit_cost),
    }

    results = {}
    for name, disp in strategies.items():
        disp  = compute_emissions(disp, mei)
        price = dam_price.reindex(disp.index)
        tc    = float(disp["charge_mw"].sum())
        td    = float(disp["discharge_mw"].sum())
        rev   = float((disp["discharge_mw"] * price).sum()
                      - (disp["charge_mw"] * price).sum())
        net_co2  = float(disp["net_em"].sum())
        fh       = marginal_fuel_hours(disp, marginal, plants)

        rec = {
            "year":              year,
            "strategy":          name,
            "total_profit_eur":  round(rev, 2),
            "total_charge_mwh":  round(tc, 3),
            "total_discharge_mwh": round(td, 3),
            "roundtrip_losses_mwh": round(tc - td, 3),
            "net_emissions_kg_co2": round(net_co2, 1),
            "net_emissions_per_mwh_charged": round(net_co2 / tc if tc > 0 else 0, 3),
            "n_cycles": round(tc / E_CAP_MWH, 1),
        }
        rec.update(fh)
        results[name] = {"kpi": rec, "dispatch": disp}
    return results




# ============================================================================
# STEP 2 - ANNUAL RUNS
# ============================================================================

def step2_annual_runs(tol=1e-6):
    print("\n" + "="*60)
    print("STEP 2 - Annual runs 2018-2025")
    print("="*60)

    amer_mix    = load_amer_mix()
    all_kpis    = []
    all_dispatch = {}
    years_ok    = []
    years_skip  = []

    prices_all = load_prices()
    dam_all    = load_dam()

    for year in range(2018, 2026):
        yfile = os.path.join(BZNL, f"meritorder_NL_{year}.csv")
        if not os.path.exists(yfile):
            years_skip.append((year, "no merit order file"))
            continue

        p_yr = prices_all.loc[f"{year}-01-01":f"{year}-12-31"]
        d_yr = dam_all.loc[f"{year}-01-01":f"{year}-12-31"]
        idx  = p_yr.index.intersection(d_yr.index)

        if len(idx) < 8000:
            years_skip.append((year, f"only {len(idx)} hours of data"))
            continue

        p_yr, d_yr = p_yr.loc[idx], d_yr.loc[idx]
        plants = load_plants(yfile)

        print(f"\n  {year}: {len(idx)} hours, {len(plants)} plants", end="", flush=True)

        srmc, marginal, mei = build_mei_for_period(plants, p_yr, d_yr, amer_mix)
        results = run_strategies(plants, p_yr, d_yr, mei, marginal, year, tol=tol)

        for strat, res in results.items():
            all_kpis.append(res["kpi"])
            out_path = os.path.join(PROC, f"annual_results_{year}_{strat}.csv")
            res["dispatch"].to_csv(out_path)

        all_dispatch[year] = {
            "mei":      mei,
            "marginal": marginal,
            "dam":      d_yr,
            "plants":   plants,
            "results":  results,
        }
        years_ok.append(year)
        print(f"  OK", flush=True)

    if years_skip:
        for yr, reason in years_skip:
            print(f"  {yr}: SKIPPED - {reason}")

    kpi_df = pd.DataFrame(all_kpis)
    kpi_df.to_csv(os.path.join(PROC, "all_annual_kpis.csv"), index=False)
    print(f"\n  Years completed: {years_ok}")
    print("  Step 2 complete.")
    return kpi_df, all_dispatch, years_ok


# ============================================================================
# STEP 3 - SUMMARY TABLES
# ============================================================================

def step3_tables(kpi_df, all_dispatch, years_ok):
    print("\n" + "="*60)
    print("STEP 3 - Summary tables")
    print("="*60)

    strats = ["profit_dp", "emission_dp", "lexico_emissions_dp", "lexico_profit_dp", "profit_greedy"]

    # Table 3A: Annual KPI summary
    rows_3a = []
    for year in years_ok:
        row = {"year": year}
        for s in strats:
            sub = kpi_df[(kpi_df["year"] == year) & (kpi_df["strategy"] == s)]
            if sub.empty:
                continue
            r   = sub.iloc[0]
            row[f"{s}_profit_eur"]          = r["total_profit_eur"]
            row[f"{s}_net_co2_kg"]          = r["net_emissions_kg_co2"]
            row[f"{s}_cycles"]              = r["n_cycles"]
            row[f"{s}_losses_mwh"]          = r["roundtrip_losses_mwh"]
            row[f"{s}_co2_per_mwh"]         = r["net_emissions_per_mwh_charged"]
        rows_3a.append(row)
    pd.DataFrame(rows_3a).to_csv(os.path.join(PROC, "table_annual_kpi_summary.csv"), index=False)
    print("  Saved: table_annual_kpi_summary.csv")

    # Table 3B: Marginal fuel hours
    fuel_cols = ["coal", "biomass", "gas", "blast_furnace_gas", "nuclear", "wind"]
    rows_3b = []
    for year in years_ok:
        row = {"year": year}
        for s in strats:
            sub = kpi_df[(kpi_df["year"] == year) & (kpi_df["strategy"] == s)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            for f in fuel_cols:
                row[f"{s}_charged_{f}_h"]    = r.get(f"hours_charged_{f}_marginal", 0)
                row[f"{s}_discharged_{f}_h"] = r.get(f"hours_discharged_{f}_marginal", 0)
        rows_3b.append(row)
    pd.DataFrame(rows_3b).to_csv(os.path.join(PROC, "table_marginal_fuel_hours.csv"), index=False)
    print("  Saved: table_marginal_fuel_hours.csv")

    # Table 3C: 2022 seasonal breakdown
    if 2022 in all_dispatch:
        rows_3c = []
        d22 = all_dispatch[2022]
        for q, (qs, qe) in enumerate(
            [("2022-01-01","2022-03-31"), ("2022-04-01","2022-06-30"),
             ("2022-07-01","2022-09-30"), ("2022-10-01","2022-12-31")], 1
        ):
            for s in strats:
                disp = d22["results"][s]["dispatch"]
                dam  = d22["dam"]
                sub  = disp.loc[qs:qe]
                price_sub = dam.reindex(sub.index)
                tc  = float(sub["charge_mw"].sum())
                td  = float(sub["discharge_mw"].sum())
                rev = float((sub["discharge_mw"] * price_sub).sum()
                            - (sub["charge_mw"] * price_sub).sum())
                rows_3c.append({
                    "quarter":  f"Q{q}",
                    "strategy": s,
                    "total_profit_eur":      round(rev, 2),
                    "total_charge_mwh":      round(tc, 3),
                    "total_discharge_mwh":   round(td, 3),
                    "roundtrip_losses_mwh":  round(tc - td, 3),
                    "net_emissions_kg_co2":  round(float(sub["net_em"].sum()), 1),
                    "n_cycles":              round(tc / E_CAP_MWH, 1),
                })
        pd.DataFrame(rows_3c).to_csv(os.path.join(PROC, "table_2022_seasonal.csv"), index=False)
        print("  Saved: table_2022_seasonal.csv")
    else:
        print("  2022 not in results - skipping seasonal table")

    print("  Step 3 complete.")


# ============================================================================
# STEP 4 - SENSITIVITY ANALYSIS
# ============================================================================

def step4_sensitivity(all_dispatch, years_ok):
    print("\n" + "="*60)
    print("STEP 4 - Sensitivity analysis (2022)")
    print("="*60)

    if 2022 not in all_dispatch:
        print("  2022 data not available - skipping sensitivity")
        return

    amer_mix = load_amer_mix()
    prices22 = load_prices().loc["2022-01-01":"2022-12-31"]
    dam22    = all_dispatch[2022]["dam"]
    plants22 = all_dispatch[2022]["plants"]
    idx22    = prices22.index.intersection(dam22.index)
    prices22, dam22 = prices22.loc[idx22], dam22.loc[idx22]

    # CO2 price sensitivity
    rows_co2 = []
    for label, co2_val in [("actual", None), ("low_15", 15.0),
                           ("medium_60", 60.0), ("high_90", 90.0)]:
        pr = prices22.copy()
        if co2_val is not None:
            pr["CO2"] = co2_val

        _, marginal, mei = build_mei_for_period(plants22, pr, dam22, amer_mix)

        for s in ["profit_dp", "emission_dp"]:
            unit = StorageUnit.from_roundtrip_efficiency(
                UNIT.p_max_mw, UNIT.e_cap_mwh, _RTE, UNIT.soc_min, UNIT.soc_max)
            cost = (UNIT.cycle_cost_eur_mwh if s == "profit_dp"
                    else UNIT.embodied_emission_kg_mwh / 1000.0)
            disp = rolling_dispatch(
                dam22 if s == "profit_dp" else mei, unit, dp, discharge_cost=cost)
            disp = compute_emissions(disp, mei)
            price_s = dam22.reindex(disp.index)
            tc  = float(disp["charge_mw"].sum())
            td  = float(disp["discharge_mw"].sum())
            rev = float((disp["discharge_mw"] * price_s).sum()
                        - (disp["charge_mw"] * price_s).sum())
            rows_co2.append({
                "co2_scenario":          label,
                "co2_eur_per_t":         co2_val if co2_val else "actual",
                "strategy":              s,
                "total_profit_eur":      round(rev, 2),
                "net_emissions_kg_co2":  round(float(disp["net_em"].sum()), 1),
            })

    pd.DataFrame(rows_co2).to_csv(
        os.path.join(PROC, "table_co2_sensitivity_2022.csv"), index=False)
    print("  Saved: table_co2_sensitivity_2022.csv")

    # Battery size sensitivity
    _, marginal_base, mei_base = build_mei_for_period(
        plants22, prices22, dam22, amer_mix)

    rows_bat = []
    for label, e_cap, p_max in [
        ("0.5h", 0.5, 1.0), ("1h",  1.0, 1.0),
        ("2h",   2.0, 1.0), ("4h",  4.0, 1.0),
    ]:
        unit = StorageUnit.from_roundtrip_efficiency(
            p_max, e_cap, _RTE, UNIT.soc_min, UNIT.soc_max)
        disp = rolling_dispatch(dam22, unit, dp, discharge_cost=UNIT.cycle_cost_eur_mwh)
        disp = compute_emissions(disp, mei_base)
        price_s = dam22.reindex(disp.index)
        tc  = float(disp["charge_mw"].sum())
        td  = float(disp["discharge_mw"].sum())
        rev = float((disp["discharge_mw"] * price_s).sum()
                    - (disp["charge_mw"] * price_s).sum())
        rows_bat.append({
            "duration_label":     label,
            "e_cap_mwh":         e_cap,
            "p_max_mw":          p_max,
            "total_profit_eur":  round(rev, 2),
            "total_charge_mwh":  round(tc, 3),
            "total_discharge_mwh": round(td, 3),
            "net_emissions_kg_co2": round(float(disp["net_em"].sum()), 1),
            "n_cycles":          round(tc / e_cap, 1),
        })

    pd.DataFrame(rows_bat).to_csv(
        os.path.join(PROC, "table_battery_size_sensitivity.csv"), index=False)
    print("  Saved: table_battery_size_sensitivity.csv")
    print("  Step 4 complete.")


# ============================================================================
# STEP 5 - FIGURES
# ============================================================================

def fuel_color_series(marginal, plants, raw_fuels=None):
    return marginal.apply(
        lambda n: FUEL_COLOR.get(fuel_of_plant(str(n), plants, raw_fuels), FUEL_COLOR["other"])
    )




def dispatch_detail(dyr, year, center_date, label, fname):
    """3-day dispatch detail centred on `center_date` (YYYY-MM-DD string).

    Three y-axes:
      Left  : charge/discharge bars + SOC - normalised to [-1, +1]
      Right inner  : DAM price  [EUR/MWh]
      Right outer  : MEI        [kg CO2/MWh_e]
    """
    ct   = pd.Timestamp(center_date)
    ft_s = str((ct - pd.Timedelta(days=1)).date())
    ft_e = str((ct + pd.Timedelta(days=1, hours=23)).date())
    mei_w = dyr["mei"].loc[ft_s:ft_e] * 1000.0
    dam_w = dyr["dam"].loc[ft_s:ft_e]

    strats_d = ["profit_dp", "emission_dp", "lexico_emissions_dp"]
    fig, axes = plt.subplots(len(strats_d), 1, figsize=(12, 9), sharex=True)
    fig.subplots_adjust(right=0.80)   # leave room for the outer right axis

    for ax, s in zip(axes, strats_d):
        disp = dyr["results"][s]["dispatch"].loc[ft_s:ft_e]

        # Left axis: bars (-1 to +1) and SOC (0-1)
        bar_w = 1 / 24
        ax.bar(disp.index,  disp["charge_mw"]    / UNIT.p_max_mw,
               color="#2166AC", alpha=0.75, width=bar_w, label="Charge [p.u.]")
        ax.bar(disp.index, -disp["discharge_mw"] / UNIT.p_max_mw,
               color="#D32F2F", alpha=0.75, width=bar_w, label="Discharge [p.u.]")
        soc_norm = disp["soc_mwh"] / UNIT.e_cap_mwh   # 0–1
        ax.plot(disp.index, soc_norm * 2 - 1,          # map 0-1 → -1 to +1
                color="black", lw=1.2, ls=":", label="SOC [0–1 → axis]")
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylim(-1.15, 1.15)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_ylabel("Charge / Discharge [-]  |  SOC [0=empty, 1=full]",
                      fontsize=8)
        ax.set_title(STRAT_LABEL[s])

        # Right inner axis: DAM price [EUR/MWh]
        ax_dam = ax.twinx()
        ax_dam.spines["right"].set_visible(True)
        ax_dam.spines["top"].set_visible(False)
        l1, = ax_dam.plot(dam_w.index, dam_w.values,
                          color="#888888", lw=1.2, ls="-", alpha=0.85,
                          label="DAM price [EUR/MWh]")
        ax_dam.set_ylabel("DAM price [EUR/MWh]", color="#555555", fontsize=8)
        ax_dam.tick_params(axis="y", colors="#555555", labelsize=7)

        # Right outer axis: MEI [kg CO2/MWh_e]
        ax_mei = ax.twinx()
        ax_mei.spines["right"].set_position(("outward", 65))
        ax_mei.spines["right"].set_visible(True)
        ax_mei.spines["top"].set_visible(False)
        l2, = ax_mei.plot(mei_w.index, mei_w.values,
                          color="#00838F", lw=1.0, ls="--", alpha=0.85,
                          label="MEI [kg CO2/MWh_e]")
        ax_mei.set_ylabel("MEI [kg CO2/MWh_e]", color="#00838F", fontsize=8)
        ax_mei.tick_params(axis="y", colors="#00838F", labelsize=7)

        # Combined legend
        h_l, l_l = ax.get_legend_handles_labels()
        ax.legend(h_l + [l1, l2], l_l + [l1.get_label(), l2.get_label()],
                  loc="upper left", fontsize=7, framealpha=0.9)

    axes[-1].set_xlabel(f"Date ({year})")
    fig.suptitle(label)
    for ext in ["png", "pdf"]:
        try:
            fig.savefig(os.path.join(FIG_DIR, f"{fname}.{ext}"))
        except PermissionError:
            pass
    plt.close(fig)
    print(f"  Saved: {fname}.png/pdf")




def step5_figures(all_dispatch, kpi_df, years_ok):
    print("\n" + "="*60)
    print("STEP 5 - Paper figures")
    print("="*60)

    strats = ["profit_dp", "emission_dp", "lexico_emissions_dp", "lexico_profit_dp", "profit_greedy"]

    # Figure 5A: Annual MEI time series 2022
    if 2022 in all_dispatch:
        d22      = all_dispatch[2022]
        mei22    = d22["mei"] * 1000.0
        dam22    = d22["dam"]
        marg22   = d22["marg"] if "marg" in d22 else d22["marginal"]
        plants22 = d22["plants"]

        # Load raw (pre-normalisation) fuel labels for granular colour mapping
        raw22    = load_raw_fuels(2022)

        # Dynamic biomass label for 2022 (50% biomass before Jul 2023)
        amer_mx  = load_amer_mix()
        bio_row  = amer_mx[amer_mx["date_from"] <= pd.Timestamp("2022-01-01")].sort_values("date_from")
        bio_pct  = int(float(bio_row.iloc[-1]["biomass_share"]) * 100) if not bio_row.empty else 0
        legend22 = [
            (f, f"Biomass ({bio_pct}%)" if f == "biomass" and bio_pct > 0 else l)
            for f, l in legend_fuels
        ]

        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax2 = ax1.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)

        # Weekly rolling for readability
        mei_week = mei22.resample("7D").mean()
        dam_week = dam22.resample("7D").mean()

        colors_w = []
        for ts in mei_week.index:
            week_marg = marg22.loc[ts:ts + pd.Timedelta(days=6)]
            if week_marg.empty:
                colors_w.append(FUEL_COLOR["other"])
                continue
            fuel_counts = week_marg.apply(
                lambda n: fuel_of_plant(str(n), plants22, raw22)
            ).value_counts()
            dom_fuel = fuel_counts.idxmax() if not fuel_counts.empty else "other"
            colors_w.append(FUEL_COLOR.get(dom_fuel, FUEL_COLOR["other"]))

        for i in range(len(mei_week) - 1):
            ax1.fill_between(
                [mei_week.index[i], mei_week.index[i+1]],
                [0, 0], [mei_week.iloc[i], mei_week.iloc[i]],
                color=colors_w[i], alpha=0.4,
            )
        ax1.plot(mei_week.index, mei_week.values, color="black", lw=0.8, alpha=0.5)
        ax2.plot(dam_week.index, dam_week.values, color="#888888",
                 lw=0.8, ls="--", alpha=0.6, label="DAM price (weekly avg)")

        ax1.set_ylabel("MEI [kg CO₂/MWh_e]  (weekly avg)")
        ax2.set_ylabel("DAM price [EUR/MWh]", color="#888888")
        ax1.set_title("Marginal Emission Intensity - Netherlands 2022")
        patches = [mpatches.Patch(color=FUEL_COLOR[f], alpha=0.7, label=l)
                   for f, l in legend22]
        ax1.legend(handles=patches, loc="upper right", fontsize=7)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(os.path.join(FIG_DIR, f"fig_mei_annual_2022.{ext}"))
        plt.close(fig)
        print("  Saved: fig_mei_annual_2022.png/pdf")

        # Figure 5B: Price-MEI scatter 2022
        fig, ax = plt.subplots(figsize=(7, 5))
        dam_c   = dam22.clip(-50, 500)
        mei_c   = (mei22).clip(0, 1100)
        colors  = fuel_color_series(marg22, plants22, raw22).reindex(dam_c.index)

        sample_idx = dam_c.index[::4]  # every 4th hour for readability
        ax.scatter(dam_c.loc[sample_idx], mei_c.loc[sample_idx],
                   c=colors.loc[sample_idx], s=3, alpha=0.4, linewidths=0)
        patches_b = [mpatches.Patch(color=FUEL_COLOR[f], label=l)
                     for f, l in legend22]
        ax.legend(handles=patches_b, fontsize=7)
        ax.set_xlabel("DAM price [EUR/MWh]")
        ax.set_ylabel("MEI [kg CO₂/MWh_e]")
        ax.set_title("Price vs MEI scatter - Netherlands 2022")
        ax.set_xlim(-50, 500); ax.set_ylim(0, 1100)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(os.path.join(FIG_DIR, f"fig_price_mei_scatter_2022.{ext}"))
        plt.close(fig)
        print("  Saved: fig_price_mei_scatter_2022.png/pdf")

    # Figure 5C: Strategy comparison annual
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(years_ok))
    width  = 0.22
    for i, s in enumerate(["profit_dp", "emission_dp", "lexico_emissions_dp", "lexico_profit_dp"]):
        vals = []
        for year in years_ok:
            sub = kpi_df[(kpi_df["year"] == year) & (kpi_df["strategy"] == s)]
            vals.append(sub["net_emissions_kg_co2"].iloc[0] / 1000.0 if not sub.empty else 0)
        ax.bar(x + i * width - width, vals, width, label=STRAT_LABEL[s],
               color=STRAT_COLOR[s], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(years_ok)
    ax.set_xlabel("Year"); ax.set_ylabel("Net CO₂ impact [tCO₂/year]")
    ax.set_title("Annual net CO₂ impact by dispatch strategy - Netherlands")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig_strategy_comparison_annual.{ext}"))
    plt.close(fig)
    print("  Saved: fig_strategy_comparison_annual.png/pdf")

    # Figure 5D: cumulative cashflow and emissions, one per year
    STRAT_LS = {
        "profit_dp":           "-",
        "emission_dp":         "-",
        "lexico_emissions_dp": "--",
        "lexico_profit_dp":    "--",
        "profit_greedy":       "-",
    }
    STRAT_LW = {
        "profit_dp":           1.8,
        "emission_dp":         1.8,
        "lexico_emissions_dp": 1.6,
        "lexico_profit_dp":    1.6,
        "profit_greedy":       1.2,
    }
    for yr in sorted(all_dispatch.keys()):
        dyr    = all_dispatch[yr]
        dam_yr = dyr["dam"]
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for s in strats:
            if s not in dyr["results"]:
                continue
            disp  = dyr["results"][s]["dispatch"]
            price = dam_yr.reindex(disp.index)
            cum_p = ((disp["discharge_mw"] * price) - (disp["charge_mw"] * price)).cumsum()
            cum_e = disp["net_em"].cumsum() / 1000.0
            ax_top.plot(cum_p.index, cum_p.values,
                        color=STRAT_COLOR[s], label=STRAT_LABEL[s],
                        lw=STRAT_LW.get(s, 1.2), ls=STRAT_LS.get(s, "-"))
            ax_bot.plot(cum_e.index, cum_e.values,
                        color=STRAT_COLOR[s], label=STRAT_LABEL[s],
                        lw=STRAT_LW.get(s, 1.2), ls=STRAT_LS.get(s, "-"))
        if yr == 2022:
            for ax in [ax_top, ax_bot]:
                ax.axvspan(pd.Timestamp("2022-08-01"), pd.Timestamp("2022-09-30"),
                           alpha=0.08, color="#D32F2F", label="Gas price spike")
        ax_top.axhline(0, color="black", lw=0.6)
        ax_bot.axhline(0, color="black", lw=0.6)
        ax_top.set_ylabel("Cumulative cashflow [EUR]")
        ax_bot.set_ylabel("Cumulative net CO2 [tCO2]")
        ax_bot.set_xlabel(str(yr))
        ax_top.set_title(f"Cumulative cashflow and emissions - Netherlands {yr}")
        ax_top.legend(fontsize=8); ax_bot.legend(fontsize=8)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            try:
                fig.savefig(os.path.join(FIG_DIR, f"fig_cumulative_{yr}.{ext}"))
            except PermissionError:
                pass
        plt.close(fig)
        print(f"  Saved: fig_cumulative_{yr}.png/pdf")


    # Figure 5E: 3-day dispatch detail (coal-charge window in 2022)
    if 2022 in all_dispatch:
        dispatch_detail(
            all_dispatch[2022], 2022, "2022-08-27",
            label="Dispatch detail - 26-28 Aug 2022  (gas crisis: coal & gas charging)",
            fname="figdispatch_detail_2022"
        )
    print("  Step 5 complete.")


# ============================================================================
# STEP 6 - GAS CRISIS ANALYSIS
# ============================================================================

def step6_gas_crisis(kpi_df, all_dispatch, years_ok):
    print("\n" + "="*60)
    print("STEP 6 - Gas crisis analysis (profit_dp by year)")
    print("="*60)

    rows = []
    for year in years_ok:
        sub = kpi_df[(kpi_df["year"] == year) & (kpi_df["strategy"] == "profit_dp")]
        if sub.empty:
            continue
        r = sub.iloc[0]

        # avg MEI during charge vs discharge
        if year in all_dispatch:
            disp = all_dispatch[year]["results"]["profit_dp"]["dispatch"]
            mei  = all_dispatch[year]["mei"] * 1000.0
            ch   = disp[disp["action"] == 1]
            dc   = disp[disp["action"] == -1]
            mei_ch = float(mei.reindex(ch.index).mean()) if len(ch) else 0
            mei_dc = float(mei.reindex(dc.index).mean()) if len(dc) else 0
        else:
            mei_ch = mei_dc = 0

        coal_ch = r.get("hours_charged_coal_marginal", 0)
        tot_ch  = sum(r.get(f"hours_charged_{t}_marginal", 0)
                      for t, _ in TRACK_FUELS) + 1e-9
        coal_pct = round(100 * coal_ch / tot_ch, 1)
        net_co2  = r["net_emissions_kg_co2"]
        direction = "added" if net_co2 > 0 else "avoided"

        rows.append({
            "year":             year,
            "net_co2_kg_profit_dp": net_co2,
            "direction":        direction,
            "coal_pct_charge":  coal_pct,
            "avg_mei_charge_kgmwh": round(mei_ch, 1),
            "avg_mei_discharge_kgmwh": round(mei_dc, 1),
        })

    df = pd.DataFrame(rows)
    print()
    print(f"  {'Year':<6} {'Net CO2 [kg]':>14} {'Direction':>8} "
          f"{'Coal% charge':>13} {'MEI charge':>11} {'MEI discharge':>14}")
    print("  " + "-"*70)
    for _, r in df.iterrows():
        print(f"  {int(r.year):<6} {r.net_co2_kg_profit_dp:>14,.0f} {r.direction:>8} "
              f"{r.coal_pct_charge:>12.1f}% {r.avg_mei_charge_kgmwh:>11.1f} "
              f"{r.avg_mei_discharge_kgmwh:>14.1f}")

    df.to_csv(os.path.join(PROC, "gas_crisis_analysis.csv"), index=False)
    print("\n  Saved: gas_crisis_analysis.csv")

    print("\n  Cross-zone analysis: Based on the Dutch results, zones with higher")
    print("  coal/lignite share in the merit order (e.g. Germany pre-2030, Poland,")
    print("  Czech Republic) would be expected to show larger positive net emissions")
    print("  under profit-maximising dispatch, as coal would be marginal more")
    print("  frequently. This is a qualitative projection for discussion only - ")
    print("  no data available for cross-zone runs.")
    print("  Step 6 complete.")
    return df


# ============================================================================
# STEP 8 - RESULTS SUMMARY
# ============================================================================

def step8_summary(kpi_df, gas_df, years_ok):
    print("\n" + "="*60)
    print("STEP 8 - RESULTS_SUMMARY.md")
    print("="*60)

    lines = [
        "# Results Summary - Battery Storage Emissions Model",
        "**Auto-generated by run_paper.py**\n",
        "---\n",
        "## Section 1: Data Availability\n",
        f"- **Years completed:** {', '.join(str(y) for y in years_ok)}",
        "- **2017:** skipped - commodity price data starts 2018-01-01",
        "- **Plant stacks:** year-specific ENTSO-E files 2017-2025",
        "- **DAM prices:** 2017-01-01 to 2026-03-12",
        "- **Commodity prices:** 2018-01-01 to 2025-12-31\n",
        "---\n",
        "## Section 2: Key Findings\n",
    ]

    if not kpi_df.empty and not gas_df.empty:
        profit_sub = kpi_df[kpi_df["strategy"] == "profit_dp"]

        if not profit_sub.empty:
            yr_max = profit_sub.loc[profit_sub["net_emissions_kg_co2"].idxmax(), "year"]
            yr_min = profit_sub.loc[profit_sub["net_emissions_kg_co2"].idxmin(), "year"]
            lines.append(f"- **Highest net emissions** (profit_dp): **{int(yr_max)}**")
            lines.append(f"- **Lowest net emissions** (profit_dp): **{int(yr_min)}**")

        # Revenue cost of emission strategy
        if len(years_ok) > 0:
            costs = []
            for year in years_ok:
                p = kpi_df[(kpi_df["year"]==year) & (kpi_df["strategy"]=="profit_dp")]
                e = kpi_df[(kpi_df["year"]==year) & (kpi_df["strategy"]=="emission_dp")]
                if not p.empty and not e.empty:
                    pp = p.iloc[0]["total_profit_eur"]
                    ep = e.iloc[0]["total_profit_eur"]
                    if abs(pp) > 1:
                        costs.append(100 * (pp - ep) / abs(pp))
            if costs:
                avg_cost = np.mean(costs)
                lines.append(f"- **Avg revenue cost** of emission strategy vs profit: "
                             f"**{avg_cost:.1f}%** of foregone profit")

        # Lexico vs emission
        lexico_better = []
        for year in years_ok:
            e = kpi_df[(kpi_df["year"]==year) & (kpi_df["strategy"]=="emission_dp")]
            l = kpi_df[(kpi_df["year"]==year) & (kpi_df["strategy"]=="lexico_emissions_dp")]
            if not e.empty and not l.empty:
                lexico_better.append(l.iloc[0]["total_profit_eur"] >= e.iloc[0]["total_profit_eur"])
        if lexico_better:
            pct = 100 * sum(lexico_better) / len(lexico_better)
            lines.append(f"- **Lexico dominates emission-only on profit** in {pct:.0f}% of years")

    lines += [
        "",
        "---\n",
        "## Section 3: Figure List\n",
        "| Figure | File | Caption |",
        "|---|---|---|",
        "| Fig. 1A | fig_schram_validation | MEI validation vs Schram et al. (2019), 11 Jan 2018 |",
        "| Fig. 1B | fig_merit_order_janSCHRAM / fig_merit_order_jan2018 | Synthetic merit order bar chart, 11 Jan 2018 hour 14 |",
        "| Fig. 5A | fig_mei_annual_2022 | Annual MEI time series Netherlands 2022 |",
        "| Fig. 5B | fig_price_mei_scatter_2022 | Price-MEI scatter plot Netherlands 2022 |",
        "| Fig. 5C | fig_strategy_comparison_annual | Annual net CO₂ by dispatch strategy |",
        "| Fig. 5D | fig_cumulative_2022 | Cumulative cashflow and emissions 2022 |",
        "| Fig. 5E | figdispatch_detail_2022 | Dispatch detail August 2022 gas crisis fortnight |\n",
        "---\n",
        "## Section 4: Table List\n",
        "| Table | File | Paper location |",
        "|---|---|---|",
        "| 3A | table_annual_kpi_summary.csv | Main results table |",
        "| 3B | table_marginal_fuel_hours.csv | Supplementary |",
        "| 3C | table_2022_seasonal.csv | Gas crisis seasonal detail |",
        "| 4A | table_co2_sensitivity_2022.csv | Sensitivity analysis |",
        "| 4B | table_battery_size_sensitivity.csv | Battery size sensitivity |",
        "| 1C | schram_comparison_11jan2018.csv | Schram validation detail |",
        "| Gas | gas_crisis_analysis.csv | Section 5 discussion |",
    ]

    md_path = os.path.join(BASE, "../RESULTS_SUMMARY.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: RESULTS_SUMMARY.md")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # Lexicographic tie tolerance [EUR]. Change here for sensitivity tests.
    # Default: 1e-6 (strict). Alternative: 0.01 (one-cent ε-lexicographic).
    _TOL = 1e-6

    legend_fuels = [
        ("coal",              "Coal (supercritical)"),
        ("biomass",           "Biomass"),
        ("gas_ccgt",          "Gas CCGT"),
        ("gas_ccgt_chp",      "Gas CCGT CHP"),
        ("gas_ocgt",          "Gas OCGT"),
        ("gas_engine_chp",    "Gas Engine CHP"),
        ("blast_furnace_gas", "Blast furnace gas"),
        ("nuclear",           "Nuclear"),
        ("res",               "RES (wind/solar)"),
    ]

    # Step 2: Annual runs
    kpi_df, all_dispatch, years_ok = step2_annual_runs(tol=_TOL)

    # Step 3: Summary tables
    step3_tables(kpi_df, all_dispatch, years_ok)

    # Step 4: Sensitivity analysis
    step4_sensitivity(all_dispatch, years_ok)

    # Step 5: Figures
    step5_figures(all_dispatch, kpi_df, years_ok)

    # Step 6: Gas crisis analysis
    gas_df = step6_gas_crisis(kpi_df, all_dispatch, years_ok)

    # Step 7: Cross-zone note (printed in step 6)

    # Step 8: Results summary
    step8_summary(kpi_df, gas_df, years_ok)

    # Final checklist
    print("\n" + "="*60)
    print("CHECKLIST")
    print("="*60)
    fig_dir = FIG_DIR
    checks = [


        ("Annual KPI table",            os.path.join(PROC, "table_annual_kpi_summary.csv")),
        ("Seasonal breakdown 2022",     os.path.join(PROC, "table_2022_seasonal.csv")),
        ("CO2 sensitivity table",       os.path.join(PROC, "table_co2_sensitivity_2022.csv")),
        ("Battery size sensitivity",    os.path.join(PROC, "table_battery_size_sensitivity.csv")),
        ("MEI annual 2022 figure",      "fig_mei_annual_2022.png"),
        ("Price-MEI scatter 2022",      "fig_price_mei_scatter_2022.png"),
        ("Strategy comparison figure",  "fig_strategy_comparison_annual.png"),
        ("Cumulative 2022 figure",      "fig_cumulative_2022.png"),
        ("Dispatch detail figure",      "figdispatch_detail_2022.png"),
        ("Gas crisis analysis",         os.path.join(PROC, "gas_crisis_analysis.csv")),
        ("RESULTS_SUMMARY.md",          os.path.join(BASE, "../RESULTS_SUMMARY.md")),
    ]
    for label, path in checks:
        full = path if os.path.isabs(path) else os.path.join(fig_dir, path)
        mark = "[x]" if os.path.exists(full) else "[ ]"
        print(f"  {mark} {label}")
