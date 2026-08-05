"""
run_lexico_tie_mechanism_check.py - Lexicographic tie diagnostics, both orders
================================================================================
Quantifies how often the two lexicographic dispatch strategies' secondary
objective actually breaks a tie in the primary objective, and whether that
ever changes the realised dispatch decision.

Four metrics per ordering (Lexico-P: profit primary, emission secondary;
Lexico-E: emission primary, profit secondary), for one calendar year:

  1. Bellman-level tie diagnostics. At every (hour, SOC-grid-point)
     evaluated in the DP backward pass, the feasible-action primary
     values are ranked; a tie is defined as a top1-vs-top2 gap <= tol
     (1e-6, in the primary signal's own units). This counts every
     backward-pass evaluation (the full Bellman recursion, ~2.6M cells
     for a full year at this grid resolution), not just the hours
     actually visited along the realised dispatch trajectory. Uses
     algorithms.dp()'s `tie_stats` instrumentation parameter - no
     changes to model/ are needed for this metric.

  2. Distinct primary-signal values. MEI is piecewise-constant (tied to
     which of the plant stack's units is marginal each hour); DAM price
     is effectively continuous. Reported as nunique() at 6-decimal
     rounding.

  3. Realised-path action differences. Compares the actual hour-by-hour
     dispatch action between the lexicographic strategy and its
     primary-only counterpart (Lexico-P vs. profit_dp; Lexico-E vs.
     emission_dp) - i.e. does the secondary tie-break objective ever
     actually change which hours the battery charges/discharges, along
     the realised path? Read directly from the already-computed
     dispatch_*.csv files in data/processed/model/.

  4. "Favorable-flip count" - not computed. This would require knowing,
     among the ties recorded in metric 1, how many occurred at a
     (hour, SOC-grid-point) that was actually visited by the forward
     pass (a tie that mattered for the realised path, as opposed to
     anywhere in the backward recursion). algorithms.dp() does not
     currently expose per-cell tie flags from its forward pass - only
     (actions, charge_mw, discharge_mw) is returned. Computing this
     rigorously would require extending dp()'s return signature with an
     optional diagnostics payload, which is a functional change to the
     core model and out of scope for this diagnostic script.

Paper-specific analysis script (not part of the distributable model) -
kept in support/, not model/. Imports the core layers from ../model/.

Run from the repository root:
    python support/run_lexico_tie_mechanism_check.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model"))

from algorithms import dp
from run import UNIT, GATE_HOUR_CET, LOOKAHEAD_HOURS
from run_paper import load_amer_mix, BASE, PROC
from run_robustness import build_year_bundle

PROFIT_COST   = UNIT.cycle_cost_eur_mwh
EMISSION_COST = UNIT.embodied_emission_kg_mwh / 1000.0  # kg/MWh -> kg/kWh
YEAR          = 2022


def bellman_tie_diagnostics(primary, secondary, discharge_cost, secondary_discharge_cost):
    """Re-run one year's rolling dispatch with the given primary/secondary
    signals, accumulating Bellman-level tie statistics via dp()'s
    tie_stats parameter. Mirrors run_final.py's lexico_p_tie_diagnostics()
    exactly, generalised to take either signal as primary."""
    tie_stats = {}
    plan = {}
    soc = 0.0
    for ts in primary.index:
        if ts.hour == GATE_HOUR_CET:
            end_ts = ts + pd.Timedelta(hours=LOOKAHEAD_HOURS - 1)
            h_pri = primary.loc[(primary.index >= ts) & (primary.index <= end_ts)]
            h_sec = secondary.reindex(h_pri.index).values
            acts, cmw, dmw = dp(
                h_pri.values, UNIT, soc,
                discharge_cost=discharge_cost,
                secondary_signal=h_sec, secondary_discharge_cost=secondary_discharge_cost,
                tie_stats=tie_stats,
            )
            for k, t in enumerate(h_pri.index):
                plan[t] = (acts[k], cmw[k], dmw[k])
        act, pc, pd_ = plan.get(ts, (0, 0.0, 0.0))
        c = d = 0.0
        if act == 1 and pc > 0:
            c = min(pc, UNIT.max_charge_mw(soc)); soc = UNIT.soc_after_charge(c, soc)
        elif act == -1 and pd_ > 0:
            d = min(pd_, UNIT.max_discharge_mw(soc)); soc = UNIT.soc_after_discharge(d, soc)
        soc = float(np.clip(soc, UNIT.soc_min_mwh, UNIT.soc_max_mwh))
    return tie_stats


def summarize_ties(label, tie_stats):
    n_dec = tie_stats.get("n_decisions", 0)
    n_ties = tie_stats.get("n_ties", 0)
    gaps = np.array(tie_stats.get("gaps", []), dtype=float)
    n_nan = int(np.isnan(gaps).sum())
    tie_rate_pct = 100.0 * n_ties / n_dec if n_dec else float("nan")
    median_gap = float(np.nanmedian(gaps)) if gaps.size else float("nan")
    return {
        "label": label,
        "n_decisions": n_dec,
        "n_ties": n_ties,
        "tie_rate_pct": tie_rate_pct,
        "median_gap": median_gap,
        "n_nan_gaps": n_nan,
        "n_gaps": gaps.size,
    }


def realised_path_diff(lexico_csv, primary_only_csv):
    a = pd.read_csv(lexico_csv, index_col=0, parse_dates=True)["action"]
    b = pd.read_csv(primary_only_csv, index_col=0, parse_dates=True)["action"]
    diff = (a != b).sum()
    return int(diff), len(a)


def main():
    print(f"\n{'='*70}\nLEXICO TIE-MECHANISM CHECK - {YEAR}\n{'='*70}\n")

    amer_mix = load_amer_mix()
    bundle = build_year_bundle(YEAR, amer_mix)
    dam, mei = bundle["dam"], bundle["mei"]

    print("Computing Bellman-level tie diagnostics (Lexico-P: profit primary) ...")
    ts_p = bellman_tie_diagnostics(dam, mei, PROFIT_COST, EMISSION_COST)
    stats_p = summarize_ties("Lexico-P (profit primary, emission secondary)", ts_p)

    print("Computing Bellman-level tie diagnostics (Lexico-E: emission primary) ...")
    ts_e = bellman_tie_diagnostics(mei, dam, EMISSION_COST, PROFIT_COST)
    stats_e = summarize_ties("Lexico-E (emission primary, profit secondary)", ts_e)

    mei_distinct = mei.round(6).nunique()
    price_distinct = dam.round(6).nunique()

    lp_path = os.path.join(PROC, "dispatch_lexico_profit_dp.csv")
    le_path = os.path.join(PROC, "dispatch_lexico_emission_dp.csv")
    pdp_path = os.path.join(PROC, "dispatch_profit_dp.csv")
    edp_path = os.path.join(PROC, "dispatch_emission_dp.csv")

    diff_p = diff_e = None
    if all(os.path.exists(p) for p in [lp_path, pdp_path]):
        diff_p = realised_path_diff(lp_path, pdp_path)
    if all(os.path.exists(p) for p in [le_path, edp_path]):
        diff_e = realised_path_diff(le_path, edp_path)

    lines = []
    lines.append(f"Lexico tie-mechanism check - {YEAR}")
    lines.append("=" * 70)
    lines.append("")

    for stats in (stats_p, stats_e):
        lines.append(stats["label"])
        lines.append("-" * len(stats["label"]))
        lines.append(f"  n_decisions (>=2 feasible actions): {stats['n_decisions']}")
        lines.append(f"  n_ties (gap <= 1e-6):                {stats['n_ties']}")
        lines.append(f"  tie_rate_pct:                        {stats['tie_rate_pct']:.4f}%")
        lines.append(f"  median_value_gap (top1-top2, all decisions): {stats['median_gap']:.6f}")
        lines.append(f"    ({stats['n_nan_gaps']} of {stats['n_gaps']} decisions excluded as NaN - "
                      f"EU DST spring-forward hour)")
        lines.append("")

    lines.append(f"Distinct primary-signal values, {YEAR}:")
    lines.append(f"  MEI (emission signal):   {mei_distinct} distinct values (6-decimal rounding)")
    lines.append(f"  DAM price (price signal): {price_distinct} distinct values (6-decimal rounding)")
    lines.append("")

    lines.append("Realised-path action differences (lexicographic vs. primary-only):")
    if diff_p is not None:
        lines.append(f"  Lexico-P vs. profit_dp:    {diff_p[0]} / {diff_p[1]} hours differ")
    else:
        lines.append("  Lexico-P vs. profit_dp:    SKIPPED (dispatch CSVs not found)")
    if diff_e is not None:
        lines.append(f"  Lexico-E vs. emission_dp:  {diff_e[0]} / {diff_e[1]} hours differ")
    else:
        lines.append("  Lexico-E vs. emission_dp:  SKIPPED (dispatch CSVs not found)")
    lines.append("")

    lines.append('"Favorable-flip count" - not computed. See module docstring: requires')
    lines.append("extending algorithms.dp()'s return signature to expose which realised")
    lines.append("(hour, SOC-grid-point) pairs were tie-broken by the secondary objective;")
    lines.append("not currently exposed, and adding it is a functional change to the core")
    lines.append("model, out of scope for this diagnostic script.")
    lines.append("")

    text = "\n".join(lines)
    print(text)

    out_path = os.path.join(BASE, "../lexico_tie_mechanism_2022.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
