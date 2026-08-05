"""
run.py - Main entry point
==========================
Wires the three layers together and produces output CSV files.

This is a reduced-form diagnostic model consisting of three separate
layers:
  Layer 1 (market.py)      Builds the market environment: reconstructs
                            the merit order and the marginal emission
                            signal from market data. Day-ahead
                            electricity prices and commodity spot prices
                            are the primary inputs.
  Layer 2 (storage.py)     Represents the storage system as a series of
                            physical and operational parameters.
  Layer 3 (algorithms.py)  The dispatch algorithm. Uses the output of the
                            first two layers to make the dispatch
                            decisions.

Execution order:
  1. Load data          (plant stack, commodity prices, DAM prices)
  2. Layer 1 - Market   build_mei_curve() → hourly SRMC, marginal plant, MEI
  3. Layer 2 - Storage  StorageUnit.from_roundtrip_efficiency() → unit
  4. Layer 3 - Dispatch rolling_dispatch() × 5 strategies:
                          a. Profit maximisation       (price signal,  DP)
                          b. Emission minimisation     (MEI signal,    DP)
                          c. Lexico-E                  (emission primary, profit secondary, DP)
                          d. Greedy benchmark          (price signal, greedy)
                          e. Lexico-P                  (profit primary, emission secondary, DP)
                          Both lexicographic orders (c, e) are included, not
                          just the emission-primary one - whether they
                          coincide is an empirical, dataset-dependent
                          question (see Strategy E's comment below), not
                          something to assume away.
  5. Emission accounting compute_emissions() per strategy
  6. Summary table       summarize() → KPI comparison
  7. Save to OUTPUT_DIR

To adapt to a different bidding zone or year, change the CONFIG section
below. No code changes elsewhere are needed.

Paper reference: Section 3, full pipeline.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# Add the model directory to the path so the three layers can be imported
# even when run.py is called from a different working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market import load_plants, build_mei_with_blends
from storage import StorageUnit
from algorithms import greedy, dp


# ===========================================================================
# CONFIGURATION - general model parameters
# Edit this section to change the simulation period, storage specification,
# or rolling-horizon settings.  Data paths are relative to this file.
# ===========================================================================

# --- Input data paths ---
PLANT_STACK_FILE = "../data/raw/BZ_NL/meritorder_NL.csv"
PRICES_FILE      = "../data/raw/BZ_NL/commodity_prices_BZNL.csv"
DAM_FILE         = "../data/raw/BZ_NL/BZ_NL.csv"
OUTPUT_DIR       = "../data/processed/model"

# --- Simulation period (inclusive, ISO 8601) ---
START_DATE = "2022-01-01"
END_DATE   = "2022-12-31"

# --- Storage unit (Table 3 in paper, "Full overview of battery storage
# system specifications") - Huawei Luna2000-2.0MWh-2H1 ---
# StorageUnit (storage.py) itself is technology-agnostic - this battery is
# just the paper's worked example, not something the code assumes. Both
# per-MWh-discharged costs below are fixed properties of this specific
# hardware model, so both live on the same UNIT object:
#   cycle_cost_eur_mwh        economic degradation proxy [EUR/MWh]
#   embodied_emission_kg_mwh  embodied GWP proxy [kg CO2eq/MWh], from the
#     Huawei LUNA2000-2.0MWH-2H1 EPD: embodied GWP excluding use stage
#     = 133.3 kg CO2eq/kWh capacity (manufacturing 111.0 + distribution
#     3.00 + installation 0.005 + end-of-life 19.3), amortised over the
#     warranted 6,570 full cycles -> 133.3 / 6570 * 1000 = 20.3 kg/MWh.
# Each dispatch call below picks whichever cost matches its signal's units
# (see algorithms.dp's `discharge_cost` parameter) - the profit objective
# uses cycle_cost_eur_mwh against the price signal, the emission objective
# uses embodied_emission_kg_mwh against the MEI signal. Sensitivity to the
# embodied estimate (0 / 13.3 / 20.3 / 25.4 kg/MWh) is reported separately
# in data/processed/model/benchmark_cem.csv (support/run_final.py) - a
# paper-specific analysis script, not part of this distributable model.
UNIT = StorageUnit.from_roundtrip_efficiency(
    p_max_mw                 = 1.016,   # rated AC power [MW]
    e_cap_mwh                = 2.032,   # usable energy capacity [MWh]
    roundtrip_efficiency     = 0.913,   # AC-AC round-trip efficiency → η_c = η_d ≈ 0.956
    soc_min                  = 0.0,     # lower SOC bound (fraction of capacity)
    soc_max                  = 1.0,     # upper SOC bound
    cycle_cost_eur_mwh       = 55.75,   # degradation proxy [EUR / MWh discharged]
    embodied_emission_kg_mwh = 20.3,    # embodied GWP proxy [kg CO2eq / MWh discharged]
)
INITIAL_SOC = 0.0   # SOC at simulation start [MWh]

# --- Rolling horizon (Section 3.4) ---
GATE_HOUR_CET   = 12   # daily re-plan at 12:00 CET (EPEX day-ahead gate closure)
LOOKAHEAD_HOURS = 36   # horizon: afternoon of current day + full next day

# ===========================================================================
# NETHERLANDS CASE STUDY - data-specific mappings and assumptions
#
# This section contains settings that are specific to the Dutch (NL) bidding
# zone dataset.  A user adapting the model to a different bidding zone should
# replace or remove everything in this section and update load_data()
# accordingly.
# ===========================================================================

# Fuel → price column mapping.
# Maps the canonical fuel keys used in market.py to column names in the
# commodity prices CSV for the NL dataset.
# Biomass is priced at the coal proxy because no open daily biomass spot-
# market series is available.  This is stated as a model limitation in the
# paper (Section 4.1).
FUEL_PRICE_MAP = {
    "gas":               "Gas",
    "coal":              "Coal",
    "biomass":           "Coal",           # proxy: coal price
    "blast_furnace_gas": "Blast_furnace_gas",
}

# ===========================================================================
# NETHERLANDS CASE STUDY - co-firing plants with time-varying fuel mix
#
# Some plants blend two fuels whose proportions change over time.  The
# blended CO₂ intensity for each plant is:
#
#     ε = Σ_fuel ( share_fuel × f_fuel ) / η
#
# where f_fuel is the thermal CO₂ factor from market.THERMAL_CO2.
#
# COFIRING_PLANTS maps each plant name (as it appears in the stack CSV) to
# the path of its fuel-mix timeline CSV.  Each timeline CSV must have:
#   date_from - start of the period (ISO date)
#   <fuel>_share - one column per blended fuel, e.g. coal_share,
#                        biomass_share.  The fuel name must be a key in
#                        market.THERMAL_CO2.
#
# To run with a simple stack (e.g. meritorder_example.csv): set to {}.
# To add another co-firing plant: add a line here and supply its CSV.
#
# Source for Amer: Global Energy Monitor, Amer power station,
#   https://www.gem.wiki/Amer_power_station, accessed 2026-05-28.
# ===========================================================================

COFIRING_PLANTS: dict[str, str] = {
    "Amer (Amercentrale)": "../data/raw/BZ_NL/amer_fuel_mix.csv",
    # "Other plant name":  "../data/raw/other_plant_mix.csv",
}


# ===========================================================================
# Data loading
# ===========================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load and align all input data.

    Returns
    -------
    plants    : pd.DataFrame  Plant stack (one row per unit).
    prices    : pd.DataFrame  Hourly commodity prices (CO2, Gas, Coal, …).
    dam_price : pd.Series     Hourly day-ahead market price [EUR / MWh_e].
    """
    base = os.path.dirname(os.path.abspath(__file__))

    # --- Plant stack ---
    # Prefer a year-specific stack (PLANT_STACK_FILE with "_<year>" inserted
    # before the extension, e.g. meritorder_NL.csv -> meritorder_NL_2022.csv)
    # matching START_DATE's year, over the generic PLANT_STACK_FILE, when one
    # exists - plant efficiencies/fuel mixes changed year to year (retirements,
    # co-firing ramp-up, etc.), so the year-specific file is the accurate
    # stack for that year. The pattern is derived from PLANT_STACK_FILE
    # itself (not hardcoded to "meritorder_NL"), so this still does the right
    # thing after adapting PLANT_STACK_FILE to a different bidding zone: it
    # looks for that zone's own per-year files and falls back to the generic
    # one when none exist. Mirrors support/run_paper.py's
    # load_data_for_year(), which the paper's canonical multi-year results
    # are built from. Only START_DATE's year is checked - a simulation period
    # spanning a year boundary uses one stack for the whole run, same as
    # load_data_for_year().
    stem, ext = os.path.splitext(PLANT_STACK_FILE)
    year = pd.Timestamp(START_DATE).year
    year_specific = f"{stem}_{year}{ext}"
    stack_file = year_specific if os.path.exists(os.path.join(base, year_specific)) else PLANT_STACK_FILE
    plants = load_plants(os.path.join(base, stack_file))

    # --- Commodity prices (daily resolution → forward-fill to hourly) ---
    prices_raw = pd.read_csv(
        os.path.join(base, PRICES_FILE),
        parse_dates=["Datum"],
        index_col="Datum",
    )
    # Rename Dutch column headers used in the NL commodity price CSV to the
    # English names expected by FUEL_PRICE_MAP.  Adjust for other datasets.
    prices_raw = prices_raw.rename(columns={
        "Kolen":             "Coal",
        "blast furnace gas": "Blast_furnace_gas",
        "solar and wind":    "Solar_wind",
        "nuclear":           "Nuclear",
    })
    # Extend daily prices to hourly by forward-filling within each day.
    hourly_idx = pd.date_range(
        start=prices_raw.index.min(),
        end=prices_raw.index.max() + pd.Timedelta(hours=23),
        freq="h",
    )
    prices = prices_raw.reindex(hourly_idx).ffill()

    # --- Validate required commodity-price columns are present ---
    # The shipped commodity_prices_BZNL.csv has the licensed CO2/Gas/Coal
    # price columns removed (see docs/DATA_ACCESS.md). Checking here fails
    # fast with an actionable message, instead of a bare KeyError raised
    # deep inside market.compute_srmc().
    required_cols = {"CO2"} | set(FUEL_PRICE_MAP.values())
    missing_cols = sorted(required_cols - set(prices.columns))
    if missing_cols:
        raise RuntimeError(
            f"Missing required commodity-price column(s) in {PRICES_FILE}: "
            f"{', '.join(missing_cols)}. The shipped CSV omits licensed price "
            "series (EU ETS CO2, TTF gas, API2 coal) for redistribution "
            "reasons - see docs/DATA_ACCESS.md for sources and reconstruction "
            "steps, and support/validate_commodity_data.py to check your "
            "rebuilt file before running the model."
        )

    # --- EPEX day-ahead prices ---
    # Parse the interval start timestamp from strings like:
    #   "01/01/2024 00:00:00 - 01/01/2024 01:00:00"        (hourly, pre-2025)
    #   "01/01/2025 00:00:00 - 01/01/2025 00:15:00"        (15-min MTU, 2025+)
    #   "26/03/2017 00:00:00 (CET) - 26/03/2017 01:00:00 (CEST)"  (DST label)
    epex_raw = pd.read_csv(os.path.join(base, DAM_FILE))
    epex_raw["datetime"] = (
        epex_raw["Date"]
        .str.split(" - ").str[0]                          # keep start timestamp
        .str.replace(r"\s*\(.*?\)", "", regex=True)       # strip "(CET)"/"(CEST)"
        .str.strip()
    )
    epex_raw["datetime"] = pd.to_datetime(epex_raw["datetime"], format="%d/%m/%Y %H:%M:%S")
    epex_raw = epex_raw.set_index("datetime").sort_index()
    dam_price = epex_raw["Price"].astype(float)

    # Resample to hourly means.  This handles two separate issues in one step:
    #
    # 1. Sub-hourly MTU data (from 2025 in this dataset).
    #    EPEX switched European DAM to 15-minute Market Time Units on
    #    2025-09-30 (delivery from 2025-10-01).  In this file, 15-minute
    #    intervals appear from 2025-01-01 onward - four rows per hour.
    #    For Jan–Sep 2025 the four values within each hour are identical
    #    (the hourly price replicated at 15-min granularity), so the mean
    #    equals the original price.  For Oct 2025 onward the four values
    #    reflect true intra-hour variation; averaging them gives a single
    #    representative hourly price.
    #    Note: extending the dispatch model to native 15-minute resolution
    #    is feasible (adjust Δt = 0.25 h throughout) but is not implemented
    #    here; the hourly formulation is retained for computational efficiency
    #    and consistency with the hourly commodity price data.
    #
    # 2. DST autumn transition: two rows share the same "02:00" timestamp
    #    after timezone labels are stripped.  Averaging both is more correct
    #    than discarding one.
    dam_price = dam_price.resample("h").mean()

    # --- Filter to simulation period ---
    prices    = prices.loc[START_DATE:END_DATE]
    dam_price = dam_price.loc[START_DATE:END_DATE]

    # --- Align on common hourly index ---
    idx       = prices.index.intersection(dam_price.index)
    prices    = prices.loc[idx]
    dam_price = dam_price.loc[idx]

    print(f"Simulation period : {idx[0]}  to  {idx[-1]}  ({len(idx)} hours)")

    return plants, prices, dam_price


# ===========================================================================
# Rolling horizon dispatch driver
# ===========================================================================

def rolling_dispatch(
    primary_signal:            pd.Series,
    unit:                      StorageUnit,
    algo,
    discharge_cost:            float = 0.0,
    secondary_signal:          pd.Series | None = None,
    secondary_discharge_cost:  float = 0.0,
    gate_hour:                 int   = GATE_HOUR_CET,
    lookahead:                 int   = LOOKAHEAD_HOURS,
    initial_soc:               float = INITIAL_SOC,
) -> pd.DataFrame:
    """Execute dispatch with daily re-planning at gate closure.

    At `gate_hour` (default 12:00) the chosen algorithm plans the next
    `lookahead` hours. The plan executes hour-by-hour with physical SOC
    accounting until the next gate. This mirrors the EPEX SPOT day-ahead
    market structure (gate closure at 12:00 CET for next-day delivery).

    Paper reference: Section 3.4 (rolling horizon / MPC structure).

    Parameters
    ----------
    primary_signal   : pd.Series    Hourly signal (price or MEI).
    unit             : StorageUnit  Storage parameters (Layer 2).
    algo             : callable     Algorithm from Layer 3 (greedy or dp).
    discharge_cost   : float        Cost subtracted from the discharge-leg
                                     reward, in the same units as
                                     `primary_signal` - e.g.
                                     unit.cycle_cost_eur_mwh for a price
                                     signal, or
                                     unit.embodied_emission_kg_mwh / 1000
                                     for an MEI signal.
    secondary_signal : pd.Series    Optional secondary signal (for DP lexico).
    secondary_discharge_cost : float  Cost for the secondary signal's own
                                     discharge leg (same formula as
                                     `discharge_cost`, just for the
                                     secondary objective).
    gate_hour        : int          Hour of day to re-plan (CET, default 12).
    lookahead        : int          Hours to look ahead from gate (default 36).
    initial_soc      : float        Starting SOC [MWh].

    Returns
    -------
    pd.DataFrame with columns:
        action         +1 charge / -1 discharge / 0 idle
        charge_mw      actual charge power [MW]
        discharge_mw   actual discharge power [MW]
        soc_mwh        state of charge at end of hour [MWh]
    """
    plan: dict[pd.Timestamp, tuple[int, float, float]] = {}
    rows = []
    soc  = float(np.clip(initial_soc, unit.soc_min_mwh, unit.soc_max_mwh))

    for ts in primary_signal.index:

        # --- Re-plan at gate hour ---
        if ts.hour == gate_hour:
            end_ts   = ts + pd.Timedelta(hours=lookahead - 1)
            h_mask   = (primary_signal.index >= ts) & (primary_signal.index <= end_ts)
            h_sig    = primary_signal.loc[h_mask]

            if len(h_sig) > 0:
                h_sec = None
                if secondary_signal is not None:
                    h_sec = secondary_signal.reindex(h_sig.index).values

                actions, c_mw_list, d_mw_list = algo(
                    h_sig.values, unit, soc,
                    discharge_cost=discharge_cost, secondary_signal=h_sec,
                    secondary_discharge_cost=secondary_discharge_cost,
                )
                for t_idx, t_ts in enumerate(h_sig.index):
                    plan[t_ts] = (actions[t_idx], c_mw_list[t_idx], d_mw_list[t_idx])

        # --- Execute the planned action for this hour ---
        action, planned_c, planned_d = plan.get(ts, (0, 0.0, 0.0))

        c_mw = 0.0
        d_mw = 0.0

        if action == 1 and planned_c > 0:
            # Charge: re-check physical feasibility against current SOC.
            c_mw = min(planned_c, unit.max_charge_mw(soc))
            soc  = unit.soc_after_charge(c_mw, soc)
        elif action == -1 and planned_d > 0:
            # Discharge: re-check physical feasibility against current SOC.
            d_mw = min(planned_d, unit.max_discharge_mw(soc))
            soc  = unit.soc_after_discharge(d_mw, soc)

        soc = float(np.clip(soc, unit.soc_min_mwh, unit.soc_max_mwh))

        if c_mw > 1e-9:
            action = 1
        elif d_mw > 1e-9:
            action = -1
        else:
            action = 0

        rows.append({
            "action":       action,
            "charge_mw":    c_mw,
            "discharge_mw": d_mw,
            "soc_mwh":      soc,
        })

    return pd.DataFrame(rows, index=primary_signal.index)


# ===========================================================================
# Emissions accounting
# ===========================================================================

def compute_emissions(dispatch: pd.DataFrame, mei: pd.Series) -> pd.DataFrame:
    """Attach marginal emission columns to a dispatch DataFrame.

    Accounting logic:
      Charging draws power from the grid, increasing demand on the marginal
      plant → adds emissions.
      Discharging supplies power to the grid, reducing demand on the marginal
      plant → avoids emissions.

    Unit conversion:
      Power [MW] × 1 h × MEI [kg / kWh_e] × 1000 [kWh / MWh] = kg CO₂

    Paper reference: Section 3 (emission accounting framework).

    Columns added
    -------------
    mei_kg_per_kwh          Marginal emission intensity at each hour.
    charge_emissions_kg     kg CO₂ attributed to charging.
    discharge_avoided_kg    kg CO₂ avoided by displacing marginal generation.
    net_emissions_kg        charge_emissions − discharge_avoided per hour.
    net_emissions_tco2_cum  Cumulative net CO₂ impact [tCO₂].
    """
    out = dispatch.copy()
    out["mei_kg_per_kwh"]         = mei.reindex(out.index)
    out["charge_emissions_kg"]    = out["charge_mw"]    * out["mei_kg_per_kwh"] * 1000.0
    out["discharge_avoided_kg"]   = out["discharge_mw"] * out["mei_kg_per_kwh"] * 1000.0
    out["net_emissions_kg"]       = out["charge_emissions_kg"] - out["discharge_avoided_kg"]
    out["net_emissions_tco2_cum"] = out["net_emissions_kg"].cumsum() / 1000.0
    return out


# ===========================================================================
# Summary table
# ===========================================================================

def summarize(
    results:   dict[str, pd.DataFrame],
    dam_price: pd.Series,
) -> pd.DataFrame:
    """Build a KPI table comparing all dispatch strategies.

    Metrics per strategy
    --------------------
    Total charged [MWh]        Electricity drawn from the grid.
    Total discharged [MWh]     Electricity injected into the grid.
    Revenue [EUR / MW]         (discharge × price) − (charge × price).
    Avg buy price [EUR/MWh]    Weighted average price when charging.
    Avg sell price [EUR/MWh]   Weighted average price when discharging.
    Net CO₂ [tCO₂ / MW·year]  Annual net emissions impact (positive = added).
    """
    rows = []
    for name, df in results.items():
        price = dam_price.reindex(df.index)
        tot_c = float(df["charge_mw"].sum())
        tot_d = float(df["discharge_mw"].sum())
        rev   = float(
            (df["discharge_mw"] * price).sum() - (df["charge_mw"] * price).sum()
        )
        avg_buy  = (
            float((df["charge_mw"] * price).sum() / tot_c) if tot_c > 0 else float("nan")
        )
        avg_sell = (
            float((df["discharge_mw"] * price).sum() / tot_d) if tot_d > 0 else float("nan")
        )
        net_co2 = float(df["net_emissions_kg"].sum() / 1000.0)

        rows.append({
            "Strategy":                  name,
            "Total charged [MWh]":       round(tot_c, 1),
            "Total discharged [MWh]":    round(tot_d, 1),
            "Revenue [EUR/MW]":          round(rev, 0),
            "Avg buy price [EUR/MWh]":   round(avg_buy, 1),
            "Avg sell price [EUR/MWh]":  round(avg_sell, 1),
            "Net CO2 [tCO2/MW/year]":    round(net_co2, 2),
        })

    return pd.DataFrame(rows).set_index("Strategy")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("Storage Emissions Model")
    print("=" * 60)

    base = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\nStep 1: loading data …")
    plants, prices, dam_price = load_data()

    # ------------------------------------------------------------------
    # 2. Layer 1 - Build the MEI curve
    # ------------------------------------------------------------------
    print("\nLayer 1: building merit order and MEI curve …")
    # Load cofiring mix timelines (if any) and build MEI with blended CO₂.
    # When COFIRING_PLANTS is empty, build_mei_with_blends degrades to
    # a plain build_mei_curve() with no overhead - safe for any plant stack.
    cofiring_mixes = {
        name: pd.read_csv(os.path.join(base, rel), comment="#",
                          parse_dates=["date_from"])
        for name, rel in (COFIRING_PLANTS or {}).items()
    }
    srmc, marginal, mei = build_mei_with_blends(
        plants, prices, dam_price, FUEL_PRICE_MAP, cofiring_mixes,
    )

    print(f"  Plants in stack : {len(plants)}")
    print(f"  Marginal fuels  : {marginal.value_counts().to_dict()}")
    print(f"  MEI range       : {mei.min():.4f} - {mei.max():.4f} kg CO2/kWh_e")

    # ------------------------------------------------------------------
    # 3. Layer 2 - Storage unit (configured in CONFIG section above)
    # ------------------------------------------------------------------
    print("\nLayer 2: defining storage unit …")
    unit = UNIT
    print(f"  P_max={unit.p_max_mw} MW  E_cap={unit.e_cap_mwh} MWh  "
          f"eta_c=eta_d={unit.eta_c:.4f}  c_cycle={unit.cycle_cost_eur_mwh} EUR/MWh  "
          f"c_em={unit.embodied_emission_kg_mwh} kg CO2eq/MWh")

    # ------------------------------------------------------------------
    # 4. Layer 3 - Dispatch
    # ------------------------------------------------------------------
    # Each strategy picks the cost matching its signal's units (Eq. 3.4.1):
    # profit runs cost the price signal in EUR/MWh; emission runs cost the
    # MEI signal in kg/kWh (embodied_emission_kg_mwh converted from kg/MWh).
    print("\nLayer 3: running dispatch strategies …")
    results: dict[str, pd.DataFrame] = {}
    profit_cost   = unit.cycle_cost_eur_mwh
    emission_cost = unit.embodied_emission_kg_mwh / 1000.0

    # Strategy A: Profit maximisation - DP on price signal.
    print(f"  A. Profit maximisation (DP, price signal, "
          f"c_cycle={unit.cycle_cost_eur_mwh} EUR/MWh) …")
    results["Profit max (DP)"] = rolling_dispatch(
        primary_signal=dam_price,
        unit=unit,
        algo=dp,
        discharge_cost=profit_cost,
    )

    # Strategy B: Emission minimisation - DP on MEI signal, embodied cost.
    print(f"  B. Emission minimisation (DP, MEI signal, "
          f"c_em={unit.embodied_emission_kg_mwh} kg/MWh) …")
    results["Emission min (DP)"] = rolling_dispatch(
        primary_signal=mei,
        unit=unit,
        algo=dp,
        discharge_cost=emission_cost,
    )

    # Strategy C: Lexicographic - MEI primary (embodied cost), price
    # secondary tie-breaker (secondary reward includes the profit cycle
    # cost, exactly as Strategy A's own reward does).
    print(f"  C. Lexico-E (DP, emission primary + profit secondary, "
          f"c_em={unit.embodied_emission_kg_mwh} kg/MWh, "
          f"c_cycle={unit.cycle_cost_eur_mwh} EUR/MWh) …")
    results["Lexico-E (emission primary)"] = rolling_dispatch(
        primary_signal=mei,
        unit=unit,
        algo=dp,
        discharge_cost=emission_cost,
        secondary_signal=dam_price,
        secondary_discharge_cost=profit_cost,
    )

    # Strategy D: Greedy benchmark - price signal, single-cycle per horizon.
    print(f"  D. Greedy benchmark (greedy, price signal, "
          f"c_cycle={unit.cycle_cost_eur_mwh} EUR/MWh) …")
    results["Profit max (greedy)"] = rolling_dispatch(
        primary_signal=dam_price,
        unit=unit,
        algo=greedy,
        discharge_cost=profit_cost,
    )

    # Strategy E: Lexicographic - profit primary (cycle cost), MEI
    # secondary tie-breaker. The mirror image of Strategy C: same dp()
    # call with primary/secondary swapped. Included alongside Strategy C
    # rather than assumed redundant, because whether the two lexicographic
    # orders coincide is an empirical, dataset- and technology-dependent
    # fact, not a structural guarantee of the algorithm. For this NL
    # 2018-2025 dataset they have been confirmed bit-identical to
    # profit_dp (price ties are rare - see support/run_paper.py's
    # lexico_p_ties_2022.txt: ~0.19% of Bellman decisions), but a
    # different bidding zone with a coarser price signal (e.g. price caps,
    # frequent scarcity pricing, or a market with far fewer distinct price
    # levels) or a different storage technology could see materially more
    # ties - at which point this strategy's result would genuinely diverge
    # from Strategy A, and dropping it would silently hide that.
    print(f"  E. Lexico-P (DP, profit primary + emission secondary, "
          f"c_cycle={unit.cycle_cost_eur_mwh} EUR/MWh, "
          f"c_em={unit.embodied_emission_kg_mwh} kg/MWh) …")
    results["Lexico-P (profit primary)"] = rolling_dispatch(
        primary_signal=dam_price,
        unit=unit,
        algo=dp,
        discharge_cost=profit_cost,
        secondary_signal=mei,
        secondary_discharge_cost=emission_cost,
    )

    # ------------------------------------------------------------------
    # 5. Emission accounting
    # ------------------------------------------------------------------
    for name in list(results.keys()):
        results[name] = compute_emissions(results[name], mei)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    summary = summarize(results, dam_price)
    print("\n--- Summary ---")
    print(summary.to_string())

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    out_dir = os.path.join(base, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # One CSV per strategy.
    safe_names = {
        "Profit max (DP)":              "dispatch_profit_dp",
        "Emission min (DP)":            "dispatch_emission_dp",
        "Lexico-E (emission primary)":  "dispatch_lexico_emission_dp",
        "Profit max (greedy)":          "dispatch_profit_greedy",
        "Lexico-P (profit primary)":    "dispatch_lexico_profit_dp",
    }
    for name, df in results.items():
        fname = safe_names.get(name, name.lower().replace(" ", "_")) + ".csv"
        df.to_csv(os.path.join(out_dir, fname))

    # MEI curve (for paper validation, Section 3.5).
    mei_df = pd.DataFrame({
        "dam_price_eur_mwh":  dam_price,
        "marginal_plant":     marginal,
        "mei_kg_per_kwh":     mei,
    })
    mei_df.to_csv(os.path.join(out_dir, "mei_curve.csv"))

    # Summary KPI table.
    summary.to_csv(os.path.join(out_dir, "summary.csv"))

    print(f"\nOutput saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
