"""
run_cem_generality_sweep.py - Embodied-intensity generality sweep (Step 4)
===========================================================================
Tests whether the emission-minimising strategy's benefit is a fragile
artefact of the specific embodied-emission estimate (c_em = 20.3 kg
CO2eq/MWh) used elsewhere in the paper, by re-solving emission_dp across a
wide range of c_em values and reporting both:

  operational_net_co2_kg      market-displacement accounting only
                               (charge_emissions - discharge_avoided),
                               independent of c_em - this is what
                               compute_emissions()/kpi() already report.
  full_accounting_net_co2_kg  operational + discharge_mwh * c_em, i.e.
                               operational net CO2 PLUS the battery's own
                               amortised manufacturing footprint charged
                               against the energy it actually delivered.
  retention_pct                100 * (-full_accounting_net_co2_kg) /
                               (-operational_net_co2_kg at c_em=0),
                               i.e. what fraction of the original (c_em=0)
                               operational emissions reduction survives
                               once (a) the DP adapts its own behaviour to
                               the higher embodied cost and (b) the full
                               lifecycle accounting is applied.

Years: 2022 (gas-crisis year) and 2024 (control/non-crisis year), so the
sweep isn't a one-year artefact either.

This is a paper-specific analysis script (not part of the distributable
model) - kept in support/, not model/.

Output: data/processed/model/cem_generality.csv only. No figures are
written here - support/produce_paper_figures.py owns all figure output
(see cem_generality_figure() there).

Run from storageemissions/:
    python support/run_cem_generality_sweep.py
"""
import os, sys, time, traceback
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model"))

from algorithms import dp
from run import UNIT
from run_paper import load_data_for_year, build_mei_for_period, load_amer_mix, rolling_dispatch, BASE, PROC
from run_robustness import kpi

YEARS   = [2022, 2024]
C_EM_SWEEP = [0.0, 13.3, 20.3, 25.4, 40.0, 60.0, 100.0, 150.0]

LOG_PATH = os.path.join(BASE, "../run_log.md")


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def safe(name, fn, *args, **kwargs):
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


def main():
    log("\n" + "=" * 70)
    log("STEP 4 - EMBODIED-INTENSITY GENERALITY SWEEP")
    log("=" * 70)

    amer_mix = load_amer_mix()
    rows = []

    for year in YEARS:
        def _run_year(year=year):
            plants, prices, dam = load_data_for_year(year)
            srmc, marginal, mei = build_mei_for_period(plants, prices, dam, amer_mix)

            year_rows = []
            baseline_operational_reduction = None  # -net_co2_kg at c_em=0

            for c_em in C_EM_SWEEP:
                cost = c_em / 1000.0  # kg/MWh -> kg/kWh
                disp = rolling_dispatch(mei, UNIT, dp, discharge_cost=cost)
                k = kpi(disp, dam, mei)

                operational_net_co2_kg = k["net_co2_kg"]
                full_accounting_net_co2_kg = operational_net_co2_kg + k["discharge_mwh"] * c_em

                if c_em == 0.0:
                    baseline_operational_reduction = -operational_net_co2_kg

                year_rows.append({
                    "year": year,
                    "c_em_kg_per_mwh": c_em,
                    "equiv_full_cycles": round(k["cycles"], 3),
                    "charge_mwh": round(k["charge_mwh"], 2),
                    "discharge_mwh": round(k["discharge_mwh"], 2),
                    "operational_net_co2_kg": round(operational_net_co2_kg, 1),
                    "full_accounting_net_co2_kg": round(full_accounting_net_co2_kg, 1),
                })

            # second pass: now that baseline is known, compute retention_pct
            for r in year_rows:
                reduction_full = -r["full_accounting_net_co2_kg"]
                r["retention_pct"] = (
                    round(100.0 * reduction_full / baseline_operational_reduction, 2)
                    if baseline_operational_reduction not in (None, 0.0)
                    else float("nan")
                )
            rows.extend(year_rows)
            return True

        safe(f"Step 4 - year {year}", _run_year)

    df = pd.DataFrame(rows)
    out_path = os.path.join(PROC, "cem_generality.csv")
    df.to_csv(out_path, index=False)
    log(f"  Saved: {out_path}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    main()
