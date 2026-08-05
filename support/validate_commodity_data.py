"""
validate_commodity_data.py - checks a rebuilt commodity_prices_BZNL.csv
=========================================================================
CO2, Gas, and Kolen are not shipped in this repository (licensed data -
see docs/DATA_ACCESS.md). After rebuilding the file from your own data
source, run this script to confirm your reconstruction matches the
dataset the paper's results were computed from: row count, date
coverage, and per-year mean/min/max for each series.

A pass does not mean byte-identical data - vendors differ slightly in
rounding and holiday-fill conventions. It means your series is close
enough (tolerance below) to reproduce the paper's numbers.

Run from storageemissions/:
    python support/validate_commodity_data.py
"""
import os
import sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "..", "data", "raw", "BZ_NL", "commodity_prices_BZNL.csv")

EXPECTED_ROWS = 2922
EXPECTED_DATE_MIN = "2018-01-01"
EXPECTED_DATE_MAX = "2025-12-31"

# Per-year mean/min/max, EUR/tCO2 (CO2) or EUR/MWh (Gas, Kolen).
# Computed from the dataset the paper's results were built from.
EXPECTED_STATS = {
    "CO2": {
        2018: (15.722,  7.07,  24.85),
        2019: (24.706, 18.35,  29.46),
        2020: (24.592, 14.60,  30.92),
        2021: (52.847, 30.92,  87.45),
        2022: (80.332, 57.91,  97.51),
        2023: (83.292, 66.49,  96.33),
        2024: (64.885, 49.50,  75.35),
        2025: (73.610, 59.76,  84.60),
    },
    "Gas": {
        2018: ( 22.253, 17.350,  29.245),
        2019: ( 14.649,  9.375,  22.995),
        2020: (  9.612,  3.510,  19.150),
        2021: ( 47.077, 15.525, 180.265),
        2022: (132.105, 69.795, 339.195),
        2023: ( 41.441, 23.105,  76.315),
        2024: ( 34.675, 22.934,  48.889),
        2025: ( 36.308, 26.602,  58.039),
    },
    "Kolen": {
        2018: (11.186,  9.18, 12.66),
        2019: ( 7.877,  6.17, 10.84),
        2020: ( 6.313,  4.99,  7.80),
        2021: (14.589,  7.49, 33.87),
        2022: (39.666, 15.10, 57.66),
        2023: (16.948, 12.46, 30.64),
        2024: (14.877, 12.49, 16.81),
        2025: (12.676, 11.03, 16.00),
    },
}

REL_TOL = 0.02  # 2% - allows for vendor rounding/fill differences


def check(label, actual, expected, tol=REL_TOL):
    if expected == 0:
        ok = abs(actual) < 1e-6
    else:
        ok = abs(actual - expected) / abs(expected) <= tol
    status = "OK" if ok else "MISMATCH"
    print(f"    [{status}] {label}: got {actual:.3f}, expected {expected:.3f}")
    return ok


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, parse_dates=["Datum"])
    print(f"Loaded {CSV_PATH}")

    all_ok = True

    print(f"\nRow count: {len(df)} (expected {EXPECTED_ROWS})")
    all_ok &= len(df) == EXPECTED_ROWS

    dmin, dmax = df["Datum"].min(), df["Datum"].max()
    print(f"Date coverage: {dmin.date()} to {dmax.date()} "
          f"(expected {EXPECTED_DATE_MIN} to {EXPECTED_DATE_MAX})")
    all_ok &= str(dmin.date()) == EXPECTED_DATE_MIN
    all_ok &= str(dmax.date()) == EXPECTED_DATE_MAX

    df["year"] = df["Datum"].dt.year

    for col, per_year in EXPECTED_STATS.items():
        if col not in df.columns:
            print(f"\n{col}: MISSING - column not found in {CSV_PATH}")
            all_ok = False
            continue
        print(f"\n{col}:")
        for year, (exp_mean, exp_min, exp_max) in per_year.items():
            rows = df.loc[df["year"] == year, col]
            if rows.isna().all():
                print(f"    [MISSING] {year}: no data")
                all_ok = False
                continue
            all_ok &= check(f"{year} mean", rows.mean(), exp_mean)
            all_ok &= check(f"{year} min",  rows.min(),  exp_min)
            all_ok &= check(f"{year} max",  rows.max(),  exp_max)

    print("\n" + "=" * 60)
    if all_ok:
        print("PASS - commodity_prices_BZNL.csv matches the expected dataset.")
    else:
        print("FAIL - see MISMATCH/MISSING lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
