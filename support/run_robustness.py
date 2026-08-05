"""
run_robustness.py - Embodied emission cost sweep (Run A) and robustness
battery (Run B) for the paper revision.

RUN A - embodied cycle cost c_em
    c_em = embodied GWP (excl. use stage) amortised over warranted full
    cycles. Central estimate: 133.3 kg CO2eq/kWh capacity / 6,570 cycles
    = 20.3 kg CO2eq/MWh discharge throughput (Huawei LUNA2000-2.0MWH-2H1
    EPD: manufacturing 111.0 + distribution 3.00 + installation 0.005 +
    end-of-life 19.3 kg CO2eq/kWh capacity).

    For emission-DP and Lexico-E (primary objective only), c_em is applied
    exactly like the profit-side cycle cost c_cycle - on the discharge leg
    only (Eq. 3.4.1), via algorithms.dp's `discharge_cost` parameter
    (c_em / 1000, converting kg/MWh to the MEI signal's kg/kWh scale).

RUN B - robustness battery (2022 crisis year + 2024 control year)
    Five stress tests on the merit-order / MEI construction. In every
    test, note that the observed day-ahead price series (`dam_price`) is
    exogenous historical data - it does not depend on our synthetic
    merit-order model. So whenever a test only perturbs SRMC/MEI
    construction (not dam_price itself), profit-DP's *dispatch decisions*
    are provably identical to the base case (same signal in, deterministic
    DP), and only the emission accounting changes. This lets most of Run B
    reuse cached profit-DP dispatch instead of re-solving the DP - the
    same technique already used for the CO2-price sensitivity in
    run_paper.py step4_sensitivity(). Only tests whose *primary DP signal*
    itself changes (negative-price MEI reassignment feeding emission-DP;
    forced-idle scarcity hours) require a fresh DP solve.

This is a paper-specific analysis script (not part of the distributable
model) - kept in support/, not model/. It imports the core layers from
../model/.

Run from storageemissions/:
    python support/run_robustness.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model"))

from market import build_mei_with_blends, per_plant_emission
from algorithms import greedy, dp
from run import UNIT, FUEL_PRICE_MAP
from run_paper import (
    load_amer_mix, load_data_for_year, build_mei_for_period,
    rolling_dispatch, compute_emissions, load_raw_fuels, display_fuel,
    BASE, PROC,
)

# ============================================================================
# CONSTANTS
# ============================================================================

E_CAP_MWH               = UNIT.e_cap_mwh                 # 2.032 MWh
C_EM_CENTRAL            = 20.3                            # kg CO2eq/MWh (central)
C_EM_LEVELS             = [0.0, 13.3, 20.3, 25.4]          # kg CO2eq/MWh sweep
WARRANTY_CYCLES_PER_YR  = 6570.0 / 10.0                    # 657 full cycles/yr

YEARS       = list(range(2018, 2026))
CRISIS_YEAR = 2022
CONTROL_YEAR = 2024

AMER_NAME = "Amer (Amercentrale)"

STRATEGIES_A = ["profit_dp", "emission_dp", "lexico_emissions_dp",
                "lexico_profit_dp", "profit_greedy"]


# ============================================================================
# YEAR DATA BUNDLE - loaded once, reused by both Run A and Run B
# ============================================================================

def build_year_bundle(year, amer_mix):
    plants, prices, dam = load_data_for_year(year)
    srmc, marginal, mei = build_mei_for_period(plants, prices, dam, amer_mix)
    raw_fuels = load_raw_fuels(year)
    mustrun_names = plants.index[
        plants["must_run_note"].notna()
        & (plants["must_run_note"].astype(str).str.strip() != "")
    ].tolist()
    return {
        "year": year, "plants": plants, "prices": prices, "dam": dam,
        "amer_mix": amer_mix, "srmc": srmc, "marginal": marginal, "mei": mei,
        "raw_fuels": raw_fuels, "mustrun_names": mustrun_names,
    }


def _cheapest_plant(srmc, names):
    """Name of the plant with lowest mean SRMC among `names` (year-mean)."""
    names = [n for n in names if n in srmc.columns]
    return srmc[names].mean(axis=0).idxmin()


def gas_ccgt_names(bundle):
    rf = bundle["raw_fuels"]
    return [n for n in rf if display_fuel(rf[n]) == "gas_ccgt" and n in bundle["srmc"].columns]


def coal_names(bundle):
    rf = bundle["raw_fuels"]
    return [n for n in rf if display_fuel(rf[n]) == "coal" and n in bundle["srmc"].columns]


# ============================================================================
# SHARED KPI HELPER
# ============================================================================

def kpi(disp, dam, mei):
    """KPIs for a dispatch trajectory against a given (dam, mei) pair.

    Only the emission accounting depends on `mei`; profit depends on `dam`.
    Passing a modified `mei` against an unmodified dispatch trajectory is
    the mechanism used throughout Run B to re-account emissions without
    re-solving the DP (see module docstring).
    """
    d = compute_emissions(disp, mei)
    price = dam.reindex(d.index)
    tc = float(d["charge_mw"].sum())
    td = float(d["discharge_mw"].sum())
    profit_eur = float((d["discharge_mw"] * price).sum() - (d["charge_mw"] * price).sum())
    net_co2_kg = float(d["net_em"].sum())
    return {
        "profit_eur": profit_eur, "net_co2_kg": net_co2_kg,
        "charge_mwh": tc, "discharge_mwh": td,
        "cycles": td / E_CAP_MWH,
    }


def sign_flip(a, b):
    return bool(np.sign(a) != np.sign(b))


# ============================================================================
# RUN A - embodied cycle cost sweep
# ============================================================================

def _row_a(year, strategy, c_em, disp, dam, mei):
    k = kpi(disp, dam, mei)
    embodied_kg = k["discharge_mwh"] * C_EM_CENTRAL
    return {
        "year": year,
        "strategy": strategy,
        "c_em": round(c_em, 2),
        "profit_eur": round(k["profit_eur"], 2),
        "net_co2_kg": round(k["net_co2_kg"], 1),
        "cycles": round(k["cycles"], 3),
        "embodied_kg": round(embodied_kg, 1),
        "co2_incl_embodied_kg": round(k["net_co2_kg"] + embodied_kg, 1),
    }, k


def run_a(year_bundles):
    print("\n" + "=" * 60)
    print("RUN A - embodied cycle cost sweep")
    print("=" * 60)

    rows = []
    base_kpi_cache = {}     # (year, strategy) -> unrounded kpi dict, c_em=0 for emission strategies
    dispatch_cache = {}     # year -> {"profit_dp": disp, "emission_dp_0": disp}

    for year in YEARS:
        b = year_bundles.get(year)
        if b is None:
            continue
        t0 = time.time()
        dam, mei = b["dam"], b["mei"]

        profit_cost   = UNIT.cycle_cost_eur_mwh
        emission_cost_central = UNIT.embodied_emission_kg_mwh / 1000.0
        disp_profit = rolling_dispatch(dam, UNIT, dp, discharge_cost=profit_cost)
        disp_lexp   = rolling_dispatch(dam, UNIT, dp, discharge_cost=profit_cost,
                                        secondary=mei, secondary_discharge_cost=emission_cost_central)
        disp_greedy = rolling_dispatch(dam, UNIT, greedy, discharge_cost=profit_cost)

        disp_em_cem0 = None
        for c_em in C_EM_LEVELS:
            emission_cost = c_em / 1000.0   # kg/MWh -> kg/kWh, matches MEI scale
            disp_em   = rolling_dispatch(mei, UNIT, dp, discharge_cost=emission_cost)
            disp_lexe = rolling_dispatch(mei, UNIT, dp, discharge_cost=emission_cost,
                                          secondary=dam, secondary_discharge_cost=profit_cost)

            row_e, k_e = _row_a(year, "emission_dp", c_em, disp_em, dam, mei)
            row_l, k_l = _row_a(year, "lexico_emissions_dp", c_em, disp_lexe, dam, mei)
            rows.append(row_e)
            rows.append(row_l)

            if c_em == 0.0:
                disp_em_cem0 = disp_em
                base_kpi_cache[(year, "emission_dp")] = k_e
                base_kpi_cache[(year, "lexico_emissions_dp")] = k_l

        row_p, k_p = _row_a(year, "profit_dp", C_EM_CENTRAL, disp_profit, dam, mei)
        row_lp, k_lp = _row_a(year, "lexico_profit_dp", C_EM_CENTRAL, disp_lexp, dam, mei)
        row_g, k_g = _row_a(year, "profit_greedy", C_EM_CENTRAL, disp_greedy, dam, mei)
        rows += [row_p, row_lp, row_g]
        base_kpi_cache[(year, "profit_dp")] = k_p
        base_kpi_cache[(year, "lexico_profit_dp")] = k_lp
        base_kpi_cache[(year, "profit_greedy")] = k_g

        dispatch_cache[year] = {"profit_dp": disp_profit, "emission_dp_0": disp_em_cem0}

        print(f"  {year}: done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    print("  Run A complete.")
    return df, base_kpi_cache, dispatch_cache


# ============================================================================
# RUN B - robustness battery variant builders
# ============================================================================

def variant_biomass_2x(b):
    """Amercentrale fuel cost priced at 2x the coal proxy (Amer only)."""
    plants = b["plants"].copy()
    prices = b["prices"].copy()
    prices["Coal_Amer2x"] = prices["Coal"] * 2.0
    plants.loc[AMER_NAME, "fuel"] = "biomass_amer2x"
    fpm = dict(FUEL_PRICE_MAP)
    fpm["biomass_amer2x"] = "Coal_Amer2x"
    mixes = {AMER_NAME: b["amer_mix"]}
    return build_mei_with_blends(plants, prices, b["dam"], fpm, mixes)


def variant_no_mustrun(b):
    """All must-run CHP / heat-led plants removed from the merit order."""
    plants = b["plants"].drop(index=[n for n in b["mustrun_names"] if n in b["plants"].index])
    mixes = {AMER_NAME: b["amer_mix"]} if AMER_NAME in plants.index else {}
    return build_mei_with_blends(plants, b["prices"], b["dam"], FUEL_PRICE_MAP, mixes)


def variant_negative_price(b, rep_gas_ccgt):
    """Negative-price hours reassigned the MEI of the cheapest gas CCGT
    plant instead of the base-case fallback (cheapest plant overall,
    typically a zero-emission renewable/nuclear unit)."""
    emission = per_plant_emission(b["plants"])
    neg_mask = b["dam"] < 0
    marginal_v = b["marginal"].copy()
    mei_v = b["mei"].copy()
    marginal_v.loc[neg_mask] = rep_gas_ccgt
    mei_v.loc[neg_mask] = float(emission[rep_gas_ccgt])
    return b["srmc"], marginal_v, mei_v


def variant_fuel_shock(b, ttf_pct, api2_pct):
    prices = b["prices"].copy()
    prices["Gas"]  = prices["Gas"]  * (1.0 + ttf_pct)
    prices["Coal"] = prices["Coal"] * (1.0 + api2_pct)
    mixes = {AMER_NAME: b["amer_mix"]} if AMER_NAME in b["plants"].index else {}
    return build_mei_with_blends(b["plants"], prices, b["dam"], FUEL_PRICE_MAP, mixes)


# ============================================================================
# RUN B - main
# ============================================================================

def run_b(year_bundles, base_kpi_cache, dispatch_cache):
    print("\n" + "=" * 60)
    print("RUN B - robustness battery")
    print("=" * 60)

    rows = []

    # ---- Test 1: BIOMASS - Amer fuel cost at 2x coal proxy -----------------
    print("  1. Biomass (Amer fuel cost 2x coal proxy) ...")
    for year in [CRISIS_YEAR, CONTROL_YEAR]:
        b = year_bundles[year]
        disp_base = dispatch_cache[year]["profit_dp"]
        _, marg_v, mei_v = variant_biomass_2x(b)
        k_base = base_kpi_cache[(year, "profit_dp")]
        k_var  = kpi(disp_base, b["dam"], mei_v)
        hours_base = int((b["marginal"] == AMER_NAME).sum())
        hours_var  = int((marg_v == AMER_NAME).sum())
        rows.append({
            "test": "biomass_2x_fuel_cost", "variant": "amer_fuel_cost_2x_coal_proxy",
            "year": year,
            "profit_eur": round(k_var["profit_eur"], 2),
            "net_co2_kg": round(k_var["net_co2_kg"], 1),
            "sign_changed": sign_flip(k_var["net_co2_kg"], k_base["net_co2_kg"]),
            "notes": (f"Amer marginal hours: base={hours_base}, variant={hours_var}; "
                      f"base net_co2_kg={round(k_base['net_co2_kg'],1)}"),
        })

    # ---- Test 2: CHP / MUST-RUN removal ------------------------------------
    print("  2. CHP/must-run removal ...")
    for year in [CRISIS_YEAR, CONTROL_YEAR]:
        b = year_bundles[year]
        disp_base = dispatch_cache[year]["profit_dp"]
        _, marg_v, mei_v = variant_no_mustrun(b)
        k_base = base_kpi_cache[(year, "profit_dp")]
        k_var  = kpi(disp_base, b["dam"], mei_v)
        chp_hours_base = int(b["marginal"].isin(b["mustrun_names"]).sum())
        rows.append({
            "test": "chp_mustrun_removed",
            "variant": f"{len(b['mustrun_names'])}_mustrun_plants_removed",
            "year": year,
            "profit_eur": round(k_var["profit_eur"], 2),
            "net_co2_kg": round(k_var["net_co2_kg"], 1),
            "sign_changed": sign_flip(k_var["net_co2_kg"], k_base["net_co2_kg"]),
            "notes": (f"base-case hours a must-run CHP unit is marginal={chp_hours_base}; "
                      f"base net_co2_kg={round(k_base['net_co2_kg'],1)}"),
        })

    # ---- Test 3: NEGATIVE PRICES --------------------------------------------
    print("  3. Negative-price MEI reassignment ...")
    for year in [CRISIS_YEAR, CONTROL_YEAR]:
        b = year_bundles[year]
        rep_gas = _cheapest_plant(b["srmc"], gas_ccgt_names(b))
        _, marg_v, mei_v = variant_negative_price(b, rep_gas)
        n_neg = int((b["dam"] < 0).sum())

        # profit-DP: dam_price unchanged -> dispatch identical, only accounting differs
        disp_base = dispatch_cache[year]["profit_dp"]
        k_base_p = base_kpi_cache[(year, "profit_dp")]
        k_var_p  = kpi(disp_base, b["dam"], mei_v)
        rows.append({
            "test": "negative_price_mei", "variant": "profit_dp", "year": year,
            "profit_eur": round(k_var_p["profit_eur"], 2),
            "net_co2_kg": round(k_var_p["net_co2_kg"], 1),
            "sign_changed": sign_flip(k_var_p["net_co2_kg"], k_base_p["net_co2_kg"]),
            "notes": (f"{n_neg} negative-price hours reassigned to {rep_gas}; "
                      f"base net_co2_kg={round(k_base_p['net_co2_kg'],1)}"),
        })

        # emission-DP: mei is the primary DP signal -> must re-solve (c_em=0,
        # matching the base_kpi_cache "emission_dp" entry from run_a)
        disp_em_v = rolling_dispatch(mei_v, UNIT, dp, discharge_cost=0.0)
        k_base_e = base_kpi_cache[(year, "emission_dp")]
        k_var_e  = kpi(disp_em_v, b["dam"], mei_v)
        rows.append({
            "test": "negative_price_mei", "variant": "emission_dp", "year": year,
            "profit_eur": round(k_var_e["profit_eur"], 2),
            "net_co2_kg": round(k_var_e["net_co2_kg"], 1),
            "sign_changed": sign_flip(k_var_e["net_co2_kg"], k_base_e["net_co2_kg"]),
            "notes": (f"{n_neg} negative-price hours reassigned to {rep_gas}; "
                      f"base net_co2_kg={round(k_base_e['net_co2_kg'],1)}"),
        })

    # ---- Test 4: SCARCITY ----------------------------------------------------
    print("  4. Scarcity-hour count + 2022 forced-idle rerun ...")
    scarcity_counts = {}
    crisis_scarcity_mask = None
    for year in YEARS:
        b = year_bundles.get(year)
        if b is None:
            continue
        max_srmc = b["srmc"].max(axis=1)
        dam_al = b["dam"].reindex(max_srmc.index)
        scarcity_mask = dam_al > max_srmc
        scarcity_counts[year] = int(scarcity_mask.sum())
        if year == CRISIS_YEAR:
            crisis_scarcity_mask = scarcity_mask

    b22 = year_bundles[CRISIS_YEAR]
    disp_scarcity = rolling_dispatch(b22["dam"], UNIT, dp, discharge_cost=UNIT.cycle_cost_eur_mwh,
                                      idle_mask=crisis_scarcity_mask)
    k_base = base_kpi_cache[(CRISIS_YEAR, "profit_dp")]
    k_var  = kpi(disp_scarcity, b22["dam"], b22["mei"])
    rows.append({
        "test": "scarcity_forced_idle", "variant": "battery_idle_in_scarcity_hours",
        "year": CRISIS_YEAR,
        "profit_eur": round(k_var["profit_eur"], 2),
        "net_co2_kg": round(k_var["net_co2_kg"], 1),
        "sign_changed": sign_flip(k_var["net_co2_kg"], k_base["net_co2_kg"]),
        "notes": (f"2022 scarcity hours (price>max modelled SRMC)={scarcity_counts[CRISIS_YEAR]}; "
                  f"all-year counts={scarcity_counts}; base net_co2_kg={round(k_base['net_co2_kg'],1)}"),
    })
    for year, cnt in scarcity_counts.items():
        rows.append({
            "test": "scarcity_hour_count", "variant": "price_gt_max_modelled_srmc",
            "year": year, "profit_eur": None, "net_co2_kg": None, "sign_changed": None,
            "notes": f"{cnt} hours",
        })

    # ---- Test 5: FUEL PRICES -------------------------------------------------
    print("  5. Fuel price shocks (TTF, API2 +/-10%) ...")
    rep_coal = _cheapest_plant(b22["srmc"], coal_names(b22))
    rep_gas22 = _cheapest_plant(b22["srmc"], gas_ccgt_names(b22))
    base_cross_hours = int((b22["srmc"][rep_coal] < b22["srmc"][rep_gas22]).sum())

    disp_base = dispatch_cache[CRISIS_YEAR]["profit_dp"]
    k_base = base_kpi_cache[(CRISIS_YEAR, "profit_dp")]

    for ttf_pct, api2_pct in [(0.10, 0.10), (0.10, -0.10), (-0.10, 0.10), (-0.10, -0.10)]:
        srmc_v, marg_v, mei_v = variant_fuel_shock(b22, ttf_pct, api2_pct)
        k_var = kpi(disp_base, b22["dam"], mei_v)
        cross_hours = int((srmc_v[rep_coal] < srmc_v[rep_gas22]).sum())
        rows.append({
            "test": "fuel_price_shock",
            "variant": f"ttf{ttf_pct:+.0%}_api2{api2_pct:+.0%}",
            "year": CRISIS_YEAR,
            "profit_eur": round(k_var["profit_eur"], 2),
            "net_co2_kg": round(k_var["net_co2_kg"], 1),
            "sign_changed": sign_flip(k_var["net_co2_kg"], k_base["net_co2_kg"]),
            "notes": (f"coal<gas_ccgt hours: base={base_cross_hours}, variant={cross_hours}; "
                      f"base net_co2_kg={round(k_base['net_co2_kg'],1)}"),
        })

    print("  Run B complete.")
    return pd.DataFrame(rows)


# ============================================================================
# MARKDOWN SUMMARY
# ============================================================================

def write_summary(run_a_df, run_b_df, out_path):
    lines = [
        "# Embodied Cycle Cost & Robustness Battery - Summary",
        "**Auto-generated by run_robustness.py**",
        "",
        "---",
        "",
        "## Run A - embodied emission cycle cost c_em",
        "",
        f"Central estimate c_em = {C_EM_CENTRAL} kg CO2eq/MWh discharge throughput "
        f"(Huawei LUNA2000-2.0MWH-2H1 EPD, 133.3 kg CO2eq/kWh capacity / 6,570 warranted "
        "full cycles). Levels tested: " + ", ".join(str(c) for c in C_EM_LEVELS) + " kg/MWh.",
        "",
    ]

    # --- cycle count vs warranty envelope ---
    lines.append("### Cycle count vs. warranty envelope (657 cycles/yr)")
    lines.append("")
    for year in sorted(run_a_df["year"].unique()):
        sub = run_a_df[(run_a_df["year"] == year)]
        prof = sub[sub["strategy"] == "profit_dp"]["cycles"]
        em0  = sub[(sub["strategy"] == "emission_dp") & (sub["c_em"] == 0.0)]["cycles"]
        prof_c = float(prof.iloc[0]) if len(prof) else float("nan")
        em0_c  = float(em0.iloc[0]) if len(em0) else float("nan")
        over_p = "OVER" if prof_c > WARRANTY_CYCLES_PER_YR else "within"
        over_e = "OVER" if em0_c > WARRANTY_CYCLES_PER_YR else "within"
        lines.append(f"- {year}: profit_dp={prof_c:.1f} cycles ({over_p} envelope), "
                     f"emission_dp(c_em=0)={em0_c:.1f} cycles ({over_e} envelope)")
    lines.append("")

    # --- fraction of c_em=0 reduction retained ---
    lines.append("### Fraction of the c_em=0 emission reduction retained, by c_em level")
    lines.append("")
    for c_em in C_EM_LEVELS[1:]:
        fracs = []
        for year in sorted(run_a_df["year"].unique()):
            sub = run_a_df[run_a_df["year"] == year]
            p  = sub[sub["strategy"] == "profit_dp"]["net_co2_kg"]
            e0 = sub[(sub["strategy"] == "emission_dp") & (sub["c_em"] == 0.0)]["net_co2_kg"]
            ec = sub[(sub["strategy"] == "emission_dp") & (sub["c_em"] == c_em)]["net_co2_kg"]
            if not (len(p) and len(e0) and len(ec)):
                continue
            baseline_reduction = float(p.iloc[0]) - float(e0.iloc[0])
            achieved_reduction = float(p.iloc[0]) - float(ec.iloc[0])
            if abs(baseline_reduction) > 1e-6:
                fracs.append(achieved_reduction / baseline_reduction)
        if fracs:
            lines.append(f"- c_em={c_em} kg/MWh: mean fraction retained across years = "
                         f"{np.mean(fracs)*100:.1f}% (range {min(fracs)*100:.1f}%-{max(fracs)*100:.1f}%)")
    lines.append("")

    # --- embodied-inclusive net CO2, all five strategies, central c_em ---
    lines.append("### Embodied-inclusive net CO2 (all 5 strategies, c_em=20.3 for embodied accounting)")
    lines.append("")
    lines.append("| Year | Strategy | Operational net CO2 [kg] | Embodied [kg] | Incl. embodied [kg] |")
    lines.append("|---|---|---:|---:|---:|")
    central_rows = run_a_df[
        ((run_a_df["strategy"].isin(["emission_dp", "lexico_emissions_dp"])) & (run_a_df["c_em"] == C_EM_CENTRAL))
        | (run_a_df["strategy"].isin(["profit_dp", "lexico_profit_dp", "profit_greedy"]))
    ]
    for _, r in central_rows.sort_values(["year", "strategy"]).iterrows():
        lines.append(f"| {int(r['year'])} | {r['strategy']} | {r['net_co2_kg']:.1f} | "
                     f"{r['embodied_kg']:.1f} | {r['co2_incl_embodied_kg']:.1f} |")
    lines.append("")

    flips = central_rows[
        (np.sign(central_rows["net_co2_kg"]) != np.sign(central_rows["co2_incl_embodied_kg"]))
        & (central_rows["net_co2_kg"] < 0)
    ]
    if not flips.empty:
        lines.append("**Sign flips (operational reduction becomes a net increase once embodied cost is included):**")
        for _, r in flips.iterrows():
            lines.append(f"- {int(r['year'])} {r['strategy']}: {r['net_co2_kg']:.1f} kg -> "
                         f"{r['co2_incl_embodied_kg']:.1f} kg")
    else:
        lines.append("No strategy/year flips sign once embodied emissions are included at c_em=20.3 kg/MWh.")
    lines.append("")

    # --- Run B ---
    lines += ["---", "", "## Run B - robustness battery", "",
              "Note: \"survives\" below is judged against the sign of net_co2_kg for the "
              "**2022 crisis-year** result (the paper's central gas-crisis finding); the 2024 "
              "control year is reported alongside for context but is not itself the survival "
              "criterion.", ""]

    def crisis_verdict(test_name, extra_filter=None):
        sub = run_b_df[(run_b_df["test"] == test_name) & (run_b_df["year"] == CRISIS_YEAR)]
        if extra_filter is not None:
            sub = sub[extra_filter(sub)]
        if sub.empty or sub["sign_changed"].isna().all():
            return "no 2022 result recorded"
        survived = not bool(sub["sign_changed"].fillna(False).any())
        return "SURVIVES" if survived else "SIGN CHANGES"

    lines.append(f"1. **Biomass (Amer 2x fuel cost):** 2022 sign - {crisis_verdict('biomass_2x_fuel_cost')}")
    lines.append(f"2. **CHP/must-run removed:** 2022 sign - {crisis_verdict('chp_mustrun_removed')}")
    lines.append(f"3. **Negative-price MEI reassignment:** 2022 sign (profit-DP) - "
                 f"{crisis_verdict('negative_price_mei', lambda s: s['variant']=='profit_dp')}; "
                 f"2022 sign (emission-DP) - "
                 f"{crisis_verdict('negative_price_mei', lambda s: s['variant']=='emission_dp')}")
    lines.append(f"4. **Scarcity forced-idle:** 2022 sign - {crisis_verdict('scarcity_forced_idle')}")
    fp = run_b_df[run_b_df["test"] == "fuel_price_shock"]
    fp_survives = not bool(fp["sign_changed"].fillna(False).any())
    lines.append(f"5. **Fuel price shocks (TTF/API2 +/-10%, 4 combos):** 2022 sign - "
                 f"{'SURVIVES in all 4 combos' if fp_survives else 'SIGN CHANGES in at least one combo'}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    t_start = time.time()
    amer_mix = load_amer_mix()

    print("Loading year bundles (plants, prices, DAM, SRMC, MEI) for 2018-2025 ...")
    year_bundles = {}
    for year in YEARS:
        t0 = time.time()
        year_bundles[year] = build_year_bundle(year, amer_mix)
        print(f"  {year}: loaded ({time.time()-t0:.0f}s)", flush=True)

    run_a_df, base_kpi_cache, dispatch_cache = run_a(year_bundles)
    run_a_path = os.path.join(PROC, "run_a_embodied.csv")
    run_a_df.to_csv(run_a_path, index=False)
    print(f"  Saved: {run_a_path}")

    run_b_df = run_b(year_bundles, base_kpi_cache, dispatch_cache)
    run_b_path = os.path.join(PROC, "run_b_robustness.csv")
    run_b_df.to_csv(run_b_path, index=False)
    print(f"  Saved: {run_b_path}")

    summary_path = os.path.join(BASE, "../EMBODIED_ROBUSTNESS_SUMMARY.md")
    write_summary(run_a_df, run_b_df, summary_path)

    print(f"\nTotal runtime: {time.time()-t_start:.0f}s")
