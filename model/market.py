"""
Layer 1 - Market
================
Builds the market environment: reconstructs the merit order and the
marginal emission intensity (MEI) signal from public market data and a
plant-level merit-order stack. Day-ahead electricity prices and
commodity spot prices (fuel, CO2) are the primary inputs to this layer.

The three steps mirror Section 3.2 of the paper exactly:

  Step 1  Load a plant stack CSV (one row per generating unit).
  Step 2  Compute each plant's Short-Run Marginal Cost (SRMC) at each hour.
  Step 3  Match the hourly day-ahead price to the merit order to identify the
          marginal plant. The CO₂ intensity of that plant is MEI(t).

The output - a time series of MEI values - is the key input to Layer 3.
It is the model's main contribution: a price-derived emission signal that
requires no exogenous emissions dataset.

Paper reference: Section 3.2
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Thermal CO₂ emission factors  [kg CO₂ / kWh_th]
# ---------------------------------------------------------------------------
# Source: EU ETS default values (Regulation EU 2018/2066, Annex II)
# Converted from kg/GJ: multiply by 3.6 / 1000.
#
# Biomass is assigned zero per Article 38(2) of the same regulation:
# biogenic CO₂ is not counted under the EU ETS.
# Nuclear and renewables have zero direct operational emissions.
#
# These factors feed into the emission intensity formula (Eq. 3.2.2):
#   ε_i = f_fuel × (1 / η_i)
THERMAL_CO2 = {
    "gas":               0.20232,   # 56.2 kg/GJ × 3.6/1000
    "coal":              0.35388,   # 98.3 kg/GJ × 3.6/1000
    "blast_furnace_gas": 0.25290,   # Schram et al. (2019) citing Afman & Wielders (2014): gas EF x 1.25
    "biomass":           0.0,       # zero under EU ETS Art. 38(2)
    "nuclear":           0.0,       # zero direct operational emissions
    "wind":              0.0,
    "solar":             0.0,
}

# ---------------------------------------------------------------------------
# Fuel label normalisation
# ---------------------------------------------------------------------------
# The plant stack CSV may use descriptive labels ("Gas CCGT", "Hard Coal").
# This table maps them to the canonical keys used throughout this module.

_FUEL_ALIASES = {
    # Natural gas - all technology variants map to canonical "gas" for SRMC
    "gas":               "gas",
    "natural gas":       "gas",
    "gas ccgt":          "gas",
    "gas chp":           "gas",
    "gas ocgt":          "gas",
    "gas ccgt chp":      "gas",   # industrial/district-heating CCGT CHP
    "gas engine chp":    "gas",   # reciprocating engine CHP (NAM Schoonebeek)
    # Coal - all grades map to canonical "coal"
    "coal":        "coal",
    "hard coal":   "coal",
    # Biomass
    "biomass":             "biomass",
    # Industrial by-product gas
    "blast furnace gas":  "blast_furnace_gas",
    "blast_furnace_gas":  "blast_furnace_gas",
    "bfg":                "blast_furnace_gas",
    # Renewables and nuclear - all zero marginal fuel cost
    "wind":              "wind",
    "solar":             "solar",
    "solar and wind":    "wind",
    "solar + wind":      "wind",
    "wind and solar":    "wind",
    "res":               "wind",
    "renewables":        "wind",
    "nuclear":           "nuclear",
}


def normalize_fuel(raw: str) -> str:
    """Map a raw fuel label to its canonical key.

    Unknown labels are returned lowercased so they can still be looked up
    if the user adds custom entries to THERMAL_CO2.
    """
    key = str(raw).strip().lower().replace("-", " ")
    key = " ".join(key.split())          # collapse multiple spaces
    return _FUEL_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Plant stack loader
# ---------------------------------------------------------------------------

def load_plants(path: str) -> pd.DataFrame:
    """Load the plant-level merit-order CSV.

    Expected columns
    ----------------
    name             : unique plant identifier
    fuel             : fuel type string (normalised by normalize_fuel)
    efficiency       : thermal efficiency as a fraction, e.g. 0.58 for 58 %
                       Also accepts percentage strings like "58%".
    var_om_eur_mwh   : variable operations & maintenance cost [EUR / MWh_e]

    Optional columns
    ----------------
    co2_kg_per_kwh   : explicit electrical CO₂ intensity [kg / kWh_e].
                       When present and non-NaN, overrides the value
                       computed from THERMAL_CO2 and efficiency. Useful
                       for setting biomass explicitly to zero without
                       relying on the THERMAL_CO2 table.
    capacity_mw      : installed capacity [MW] - retained for reference,
                       not used in dispatch.

    Returns a DataFrame indexed by plant name.
    """
    plants = pd.read_csv(path)
    plants.columns = [c.strip() for c in plants.columns]

    # --- Normalise efficiency ---
    # Accept both "0.58" and "58%" formats.
    eff = plants["efficiency"].astype(str).str.strip().str.rstrip("%").astype(float)
    plants["efficiency"] = np.where(eff > 1.0, eff / 100.0, eff)

    # --- Normalise fuel labels ---
    plants["fuel"] = plants["fuel"].apply(normalize_fuel)

    plants = plants.set_index("name")
    return plants


# ---------------------------------------------------------------------------
# SRMC calculation  (Section 3.2, Equation 3.2.1)
# ---------------------------------------------------------------------------

def compute_srmc(
    plants: pd.DataFrame,
    prices: pd.DataFrame,
    fuel_price_map: dict,
) -> pd.DataFrame:
    """Compute each plant's Short-Run Marginal Cost (SRMC) at every hour.

    Equation 3.2.1 (paper):
        SRMC_i = p_fuel,i / η_i  +  c_CO₂ × ε_i  +  c_OM,i

    where:
        p_fuel,i  fuel price              [EUR / MWh_th]
        η_i       thermal efficiency      [dimensionless]
        c_CO₂     EU ETS carbon price     [EUR / tCO₂]
        ε_i       CO₂ emission intensity  [tCO₂ / MWh_e]  (see Eq. 3.2.2)
        c_OM,i    variable O&M cost       [EUR / MWh_e]

    Emission intensity is derived from the fuel's thermal CO₂ factor f_fuel
    and the plant's heat rate (Equation 3.2.2):
        ε_i = f_fuel × (1 / η_i)

    Unit check on the CO₂ cost term:
        c_CO₂ [EUR/t] × ε_i [kg/kWh_e]
        = EUR/t × (1 t / 1000 kg) × (1000 kWh / 1 MWh)
        = EUR/t × t/MWh_e
        = EUR/MWh_e  ✓   (because 1 kg/kWh = 1 t/MWh numerically)

    Parameters
    ----------
    plants : pd.DataFrame
        Plant stack loaded by load_plants().
    prices : pd.DataFrame
        Time-indexed DataFrame with columns:
          'CO2'   EU ETS carbon price  [EUR / tCO₂]
          + one column per fuel type as specified by fuel_price_map.
    fuel_price_map : dict
        Maps canonical fuel key → column name in prices.
        Example: {"gas": "Gas", "coal": "Coal", "biomass": "Coal"}
        Fuels absent from this map are treated as zero marginal fuel cost.

    Returns
    -------
    srmc : pd.DataFrame
        Time-indexed, one column per plant [EUR / MWh_e].

    Notes
    -----
    Plants with a co-firing or blended fuel mix should have their effective
    co2_kg_per_kwh set in the plant stack DataFrame before calling this
    function.  The meritorder CSV accepts an explicit co2_kg_per_kwh column
    for this purpose; case-study-specific pre-processing should write the
    correct blended value there (see run.py for the Netherlands example).
    """
    srmc = pd.DataFrame(index=prices.index)

    for name, plant in plants.iterrows():
        fuel      = str(plant["fuel"])
        eta       = float(plant["efficiency"])
        var_om    = float(plant["var_om_eur_mwh"])
        heat_rate = 1.0 / eta              # MWh_th consumed per MWh_e produced

        # --- CO₂ intensity [kg CO₂ / kWh_e] (= t CO₂ / MWh_e numerically) ---
        # Use the explicit column if present; otherwise derive from THERMAL_CO2.
        if "co2_kg_per_kwh" in plant.index and not pd.isna(plant["co2_kg_per_kwh"]):
            co2_intensity = float(plant["co2_kg_per_kwh"])
        else:
            # Eq. 3.2.2: ε_i = f_fuel × (1 / η_i)
            co2_intensity = THERMAL_CO2.get(fuel, 0.0) * heat_rate

        # --- Fuel cost [EUR / MWh_e] ---
        price_col = fuel_price_map.get(fuel)
        if price_col is not None and price_col in prices.columns:
            fuel_cost = prices[price_col] * heat_rate
        else:
            fuel_cost = 0.0          # zero-marginal-cost fuels (wind, solar, nuclear)

        co2_cost = prices["CO2"] * co2_intensity
        srmc[name] = fuel_cost + co2_cost + var_om

    return srmc


# ---------------------------------------------------------------------------
# Marginal plant identification
# ---------------------------------------------------------------------------

def find_marginal_plant(srmc: pd.DataFrame, dam_price: pd.Series) -> pd.Series:
    """Identify the marginal generating unit at each hour.

    The marginal plant is the most expensive unit that is still at or below
    the market price - the last unit dispatched before market clearing.

    If the market price falls below the entire merit stack (e.g. during
    negative prices), the cheapest plant is returned as the fallback.

    Paper reference: Section 3.2 (merit-order crossing rule).

    Parameters
    ----------
    srmc : pd.DataFrame   Per-plant hourly SRMC [EUR / MWh_e].
    dam_price : pd.Series Hourly day-ahead price [EUR / MWh_e].

    Returns
    -------
    marginal : pd.Series  Plant name for each hour.
    """
    # Align to the common time index.
    idx = srmc.index.intersection(dam_price.index)
    srmc      = srmc.loc[idx]
    dam_price = dam_price.loc[idx]

    marginal = pd.Series(index=idx, dtype=object)

    for t, price in dam_price.items():
        plant_costs = srmc.loc[t]
        below = plant_costs[plant_costs <= price]
        if not below.empty:
            marginal.loc[t] = below.idxmax()       # most expensive at or below price
        else:
            marginal.loc[t] = plant_costs.idxmin() # fallback: cheapest in stack

    return marginal


# ---------------------------------------------------------------------------
# Helper: per-plant CO2 intensity lookup
# ---------------------------------------------------------------------------

def per_plant_emission(plants: pd.DataFrame) -> dict:
    """Return {plant_name: CO2 intensity [kg CO2 / kWh_e]} for every plant.

    Uses the explicit `co2_kg_per_kwh` column when present (e.g. blended
    co-firing plants), otherwise derives it from THERMAL_CO2 and efficiency
    (Eq. 3.2.2). Factored out of build_mei_curve so callers building custom
    MEI variants (e.g. counterfactual marginal-plant overrides) can reuse
    the same per-plant intensity lookup.
    """
    emission = {}
    for name, plant in plants.iterrows():
        fuel      = str(plant["fuel"])
        heat_rate = 1.0 / float(plant["efficiency"])
        if "co2_kg_per_kwh" in plant.index and not pd.isna(plant["co2_kg_per_kwh"]):
            emission[name] = float(plant["co2_kg_per_kwh"])
        else:
            emission[name] = THERMAL_CO2.get(fuel, 0.0) * heat_rate
    return emission


# ---------------------------------------------------------------------------
# Full Layer 1 pipeline
# ---------------------------------------------------------------------------

def build_mei_curve(
    plants: pd.DataFrame,
    prices: pd.DataFrame,
    dam_price: pd.Series,
    fuel_price_map: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Run the full Layer 1 pipeline and return the MEI curve.

    Calls compute_srmc → find_marginal_plant → maps to CO₂ intensity.

    MEI(t) = ε_{i*(t)} - the CO₂ intensity of the marginal plant at hour t.

    Any case-study-specific pre-processing of the plant stack (e.g. setting
    blended co2_kg_per_kwh values for co-firing plants) should be applied to
    `plants` before calling this function.

    Parameters
    ----------
    plants        : pd.DataFrame  Plant stack (output of load_plants, optionally
                                  enriched by case-study pre-processing).
    prices        : pd.DataFrame  Hourly commodity prices (CO2, Gas, Coal, …).
    dam_price     : pd.Series     Hourly day-ahead market price [EUR / MWh_e].
    fuel_price_map: dict          Fuel → price column mapping.

    Returns
    -------
    srmc      : pd.DataFrame  Per-plant SRMC [EUR / MWh_e].
    marginal  : pd.Series     Marginal plant name per hour.
    mei       : pd.Series     Marginal emission intensity [kg CO₂ / kWh_e].
    """
    # Build per-plant emission intensities (constant over time).
    # For plants whose co2_kg_per_kwh was set by case-study pre-processing,
    # that value is used directly; all others derive from THERMAL_CO2.
    emission = per_plant_emission(plants)

    srmc     = compute_srmc(plants, prices, fuel_price_map)
    marginal = find_marginal_plant(srmc, dam_price)

    # MEI(t) = ε of the marginal plant - one vectorised lookup.
    mei = marginal.map(emission).astype(float)
    mei.name = "MEI_kg_per_kWh"
    return srmc, marginal, mei


# ---------------------------------------------------------------------------
# Co-firing / blended-fuel extension  (Layer 1 add-on)
# ---------------------------------------------------------------------------

def apply_cofiring_blend(
    plants:       pd.DataFrame,
    plant_name:   str,
    period_start: pd.Timestamp,
    mix:          pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy of ``plants`` with the blended CO₂ intensity for one plant.

    Reads every ``*_share`` column in ``mix``, looks up the thermal CO₂ factor
    for each fuel in ``THERMAL_CO2``, and writes the resulting blended
    ``co2_kg_per_kwh`` into the plant stack.  Works for any fuel pair whose
    canonical keys exist in ``THERMAL_CO2``.

    Parameters
    ----------
    plants       : Plant stack indexed by plant name (output of load_plants).
    plant_name   : Index label of the co-firing plant in ``plants``.
    period_start : Timestamp of the sub-period start; selects the applicable
                   row from ``mix`` (most recent ``date_from`` ≤ period_start;
                   ``mix`` need not be pre-sorted, this function sorts it).
    mix          : Fuel-mix timeline DataFrame with columns ``date_from`` and
                   one ``<fuel>_share`` column per blended fuel.  The fuel name
                   (without ``_share``) must be a key in ``THERMAL_CO2``.

    Returns
    -------
    pd.DataFrame - copy of ``plants`` with updated ``co2_kg_per_kwh`` for the
                   named plant.  Returns ``plants`` unchanged if the plant is not
                   found or no mix row qualifies.

    Example
    -------
    For Amer in 2024 with coal_share=0.2, biomass_share=0.8 and η=0.436:
        blended_thermal = 0.2 × 0.35388 + 0.8 × 0.0 = 0.07078 kg CO₂/kWh_th
        blended_co2     = 0.07078 / 0.436            = 0.162  kg CO₂/kWh_e
    """
    if plant_name not in plants.index:
        return plants

    # Sort defensively rather than trust the caller - selecting "the most
    # recent qualifying row" via .iloc[-1] is only correct if sorted
    # ascending by date_from, and a caller loading the mix CSV without
    # sorting it would otherwise silently get the wrong blend.
    eligible = mix[mix["date_from"] <= period_start].sort_values("date_from")
    if eligible.empty:
        return plants
    row = eligible.iloc[-1]

    blended_thermal = sum(
        float(row[col]) * THERMAL_CO2.get(col.replace("_share", ""), 0.0)
        for col in mix.columns
        if col.endswith("_share")
    )
    eta = float(plants.loc[plant_name, "efficiency"])

    plants = plants.copy()
    plants.loc[plant_name, "co2_kg_per_kwh"] = blended_thermal / eta
    return plants


def build_mei_with_blends(
    plants:          pd.DataFrame,
    prices:          pd.DataFrame,
    dam_price:       pd.Series,
    fuel_price_map:  dict,
    cofiring_mixes:  dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Run the full Layer 1 pipeline with time-varying co-firing blends.

    Extends ``build_mei_curve`` to handle plants whose fuel mix changes over
    time (e.g. a coal plant gradually converting to biomass co-firing).  The
    simulation window is split at every mix-transition boundary, and each
    sub-period is run with the correct blended CO₂ intensity.

    If ``cofiring_mixes`` is empty or None, this function is identical to
    ``build_mei_curve`` with no additional overhead.

    Parameters
    ----------
    plants         : Plant stack (output of load_plants).
    prices         : Hourly commodity prices (CO2, Gas, Coal, …).
    dam_price      : Hourly day-ahead market price [EUR / MWh_e].
    fuel_price_map : Fuel → price-column mapping (same as build_mei_curve).
    cofiring_mixes : ``{plant_name: mix_df}`` where mix_df is a DataFrame with
                     columns ``date_from`` and ``<fuel>_share`` columns.
                     Each entry triggers a mid-period blend update for that plant.

    Returns
    -------
    srmc     : pd.DataFrame  Per-plant SRMC [EUR / MWh_e].
    marginal : pd.Series     Marginal plant name per hour.
    mei      : pd.Series     MEI [kg CO₂ / kWh_e].
    """
    if not cofiring_mixes:
        return build_mei_curve(plants, prices, dam_price, fuel_price_map)

    sim_start = prices.index.min()
    sim_end   = prices.index.max()

    # Collect every transition boundary that falls inside the window.
    inner_bounds: set = set()
    for mx in cofiring_mixes.values():
        for bd in mx["date_from"]:
            if sim_start < bd <= sim_end:
                inner_bounds.add(bd)

    splits = sorted([sim_start] + list(inner_bounds))

    results = []
    for i, sp in enumerate(splits):
        nxt  = splits[i + 1] if i + 1 < len(splits) else None
        mask = (
            (prices.index >= sp) & (prices.index < nxt)
            if nxt is not None
            else prices.index >= sp
        )
        sp_prices = prices.loc[mask]
        sp_dam    = dam_price.loc[mask]
        if sp_prices.empty:
            continue

        sp_plants = plants
        for pname, mx in cofiring_mixes.items():
            sp_plants = apply_cofiring_blend(sp_plants, pname, sp, mx)

        results.append(build_mei_curve(sp_plants, sp_prices, sp_dam, fuel_price_map))

    if len(results) == 1:
        results[0][2].name = "MEI_kg_per_kWh"
        return results[0]

    srmc     = pd.concat([r[0] for r in results])
    marginal = pd.concat([r[1] for r in results])
    mei      = pd.concat([r[2] for r in results])
    mei.name = "MEI_kg_per_kWh"
    return srmc, marginal, mei
