# Data Provenance

Which raw data feeds the model, exactly how each file is used, and where
to look if a number seems wrong.

## 1. Raw data inventory

All raw inputs live under `data/raw/BZ_NL/` (Dutch bidding zone, BZN|NL),
except two root-level reference files noted at the end.

| File | Content | Date range | Role |
|---|---|---|---|
| `meritorder_NL.csv` | Generic (year-agnostic) plant stack: 32 units, one row per generator, with fuel, efficiency, var. O&M, capacity, source citations | n/a (static) | **Fallback only** - used when no year-specific file exists for `START_DATE`'s year |
| `meritorder_NL_2018.csv` … `meritorder_NL_2025.csv` | Year-specific plant stacks - same columns, but reflecting that year's actual fleet (retirements/commissions applied) | one file per year | **Primary plant stack** for every year 2018-2025 (see §3) |
| `plant_events.csv` | Commissioning/decommissioning dates, one row per event, each with a citation | events from ~2012 onward | Source record for *why* each yearly stack differs from the next (not read by the model directly - informs how the yearly CSVs above were compiled) |
| `commodity_prices_BZNL.csv` | `Datum` (date) and `blast furnace gas` (present for schema compatibility, zero for the full period - BFG-fired plants carry zero fuel cost in the SRMC, only CO2 cost and var O&M). The `CO2`, `Gas`, and `Kolen` (coal) columns used in the paper are licensed and not shipped - see `docs/DATA_ACCESS.md` for sourcing and reconstruction | 2018-01-01 to 2025-12-31, daily (2,922 rows) | Fuel/carbon cost input to the SRMC formula (Eq. 3.2.1) |
| `BZ_NL.csv` | EPEX day-ahead hourly price, Netherlands | 2017-01-01 to 2026-03 (111,984 rows; only the requested year's slice is used) | The market-clearing price signal - everything in Layer 1 (marginal-plant identification) and Layer 3 (dispatch decisions) is driven by this series |
| `amer_fuel_mix.csv` | Coal/biomass co-firing share timeline for the Amer plant, 4 transition periods from 1900 (all-coal baseline) to 2025 (full biomass) | n/a (transition dates) | Blends Amer's CO2 intensity between coal and biomass shares, applied via `apply_cofiring_blend()` |
| `data/raw/meritorder_example.csv` | Toy 2-plant stack | n/a | Illustrative only, referenced in a `run.py` comment for zone-adaptation guidance - never loaded by default |

Per-plant sourcing (capacity, efficiency, confidence level, citation) is
documented exhaustively in **`docs/PLANT_SOURCES.md`** - that file is the
authority on *why* each plant has the parameters it does; this document
only covers how those parameters flow through the pipeline.

## 2. How the data is used - Layer 1 (market.py), step by step

For a given year's run (`model/run.py::load_data()` and
`support/run_paper.py::load_data_for_year()` both follow this same
sequence):

1. **Plant stack loaded** (`market.load_plants()`): reads the year-specific
   `meritorder_NL_{year}.csv` (falling back to the generic
   `meritorder_NL.csv` only if that year's file doesn't exist). Efficiency
   is normalized to a 0-1 fraction; fuel labels are canonicalized via
   `normalize_fuel()` (e.g. "Gas CCGT CHP" → `gas`).

2. **Commodity prices loaded and forward-filled**: `commodity_prices_BZNL.csv`
   is daily; each day's value is forward-filled to all 24 hours of that day
   before use, since SRMC needs an hourly series.

3. **DAM prices loaded**: `BZ_NL.csv`'s interval-labeled rows (e.g.
   `"01/01/2022 00:00:00 - 01/01/2022 01:00:00"`) are parsed to hourly
   timestamps and resampled to hourly means (handles both the pre-2025
   hourly format and the 15-minute MTU format EPEX switched to from 2025).

4. **Amer co-firing blend applied** (`apply_cofiring_blend()`): for the one
   plant with a time-varying fuel mix (Amer), the applicable row in
   `amer_fuel_mix.csv` (most recent `date_from` ≤ the period start) is used
   to compute a blended `co2_kg_per_kwh` = weighted average of coal's and
   biomass's thermal CO2 factors, divided by Amer's efficiency. This
   overwrites Amer's CO2 intensity in the plant stack for that sub-period
   only; every other plant uses its fixed intensity for the whole year.

5. **SRMC computed per plant, per hour** (`compute_srmc()`, Eq. 3.2.1):
   `fuel_price / efficiency + CO2_price × emission_intensity + var_OM`.
   The fuel price column used for each fuel is fixed by `FUEL_PRICE_MAP`
   in `run.py`:
   - `gas` → `Gas` column
   - `coal` → `Coal` column
   - `biomass` → **`Coal` column** (a deliberate proxy - no independent daily
     biomass price series exists; stated as a model limitation in the paper)
   - `blast_furnace_gas` → `Blast_furnace_gas` column (zero for the full
     period, see §1)
   - `nuclear`, `wind`/`solar`/`res` → no price column (zero marginal fuel cost)

6. **Marginal plant identified per hour** (`find_marginal_plant()`): for
   each hour, the most expensive plant whose SRMC is still at or below the
   actual EPEX day-ahead price is the marginal plant (merit-order crossing
   rule). If the price is below every plant's SRMC (can happen at negative
   prices), the cheapest plant in the stack is used as a fallback.

7. **MEI(t) read off** as that marginal plant's CO2 intensity - either the
   value implied by `THERMAL_CO2[fuel] / efficiency`, or an explicit
   `co2_kg_per_kwh` override in the plant-stack CSV (used for Amer's
   blended value from step 4).

Steps 1-7 are what produces every `mei_curve.csv` and every
`annual_results_{year}_{strategy}.csv` file under `data/processed/model/`.

## 3. Which plant stack file is used per year

`load_data()`/`load_data_for_year()` prefer the year-specific stack file
and fall back to the generic `meritorder_NL.csv` only if that year's file
is missing. For 2018-2025, a year-specific file exists for every year, so
the fallback is never invoked:

| Year | Plants in stack | Hours |
|---|---|---|
| 2018 | 33 | 8,760 |
| 2019 | 33 | 8,760 |
| 2020 | 32 | 8,784 (leap year) |
| 2021 | 32 | 8,760 |
| 2022 | 32 | 8,760 |
| 2023 | 31 | 8,760 |
| 2024 | 30 | 8,784 (leap year) |
| 2025 | 30 | 8,760 |

The declining plant count (33→30) reflects real retirements over the
period (per `plant_events.csv`), e.g. coal units retiring under the Dutch
Coal Phase-Out Act.

## 4. Layer 2 (storage) and Layer 3 (dispatch) - what's not data-driven

Once Layer 1 produces the hourly (SRMC, marginal plant, MEI, DAM price)
series, everything downstream is deterministic given the `StorageUnit`
parameters (`model/run.py`'s `UNIT`) and the dispatch algorithm (`greedy`
or `dp` from `model/algorithms.py`) - no additional raw data is consulted.
If an output number looks wrong, the two places to check are (a) did
Layer 1 compute the right MEI/price series for that hour (traceable to
the raw files in §1 via the steps in §2), or (b) is the dispatch decision
itself correct given that series.

## 5. Known data-quality caveats

- **Biomass priced at the coal proxy** (FUEL_PRICE_MAP, §2 step 5) - no
  independent daily biomass spot price exists. Affects Amer's SRMC in
  proportion to its biomass share (which itself increases over time per
  `amer_fuel_mix.csv`).
- **Several plants have ESTIMATE-confidence efficiency** (no primary
  source) - see `docs/PLANT_SOURCES.md` §3 for the full list (Amer, Elsta,
  Moerdijk 1, Rijnmond 1, Diemen 33, Delesto 2, Lage Weide 6, NAM
  Schoonebeek, Den Haag, RoCa, Pergen 1/2, Swentibold, Bergum, Eems 20,
  Borssele). These plants' SRMC - and therefore any hour where one of them
  is marginal - carries more uncertainty than PRIMARY-sourced plants.
- **Heat-led / must-run CHP plants are included in the merit order as an
  approximation** (`docs/PLANT_SOURCES.md` §2) - their real-world dispatch
  is driven by industrial steam contracts, not electricity price, so
  treating them as pure price-takers is a simplification, flagged in the
  paper.
- **Moerdijk 1 CHP's operational status is uncertain 2021-2025** -
  capacity is set to 0 MW in the historical stacks from 2018 pending
  verification against ENTSO-E Transparency data.
- **`BZ_NL.csv` spans 2017 to March 2026**, well beyond the 2018-2025 study
  window - only the requested year's slice is ever used; the extra data is
  harmless but worth knowing about if inspecting the raw file directly.

## 6. If you want to spot-check a specific number

1. Pick the year and strategy (e.g. 2022, `emission_dp`).
2. Open `data/processed/model/annual_results_2022_emission_dp.csv` (or
   `mei_curve.csv` for the underlying MEI series) and find the hour of
   interest.
3. Cross-reference that hour's `marginal_plant` against `meritorder_NL_2022.csv`
   for its fuel/efficiency, and against `docs/PLANT_SOURCES.md` for that
   plant's source citation and confidence level.
4. Cross-reference that hour's commodity prices against
   `commodity_prices_BZNL.csv` (remembering the daily→hourly forward-fill)
   and the DAM price against `BZ_NL.csv`.
5. Recompute SRMC by hand with Eq. 3.2.1 (§2 step 5 above) to confirm the
   marginal-plant call was correct for that hour.
